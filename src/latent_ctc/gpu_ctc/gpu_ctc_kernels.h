#pragma once

#include "contrib/moderngpu/include/device/ctascan.cuh"
#include "contrib/moderngpu/include/device/ctamerge.cuh"

#include "ctc_helper.h"

/*
这份代码实现的是 GPU 上的 CTC 前向、后向与梯度计算。整体逻辑是：
1. 每个 CUDA block 负责一个样本。
2. 先在 compute_alpha_kernel 中把原始标签扩展成 blank-label 交替形式。
3. 在 log 空间中逐帧递推 alpha 表。
4. 在最后一帧对右端可接受状态做 logsumexp，得到 log P(y|x)。
5. compute_betas_and_grad_kernel 从最后一帧反向递推 beta。
6. 每个时间步把 alpha + beta 得到状态级路径后验。
7. 因为同一个输出类别可能对应多个 CTC 状态，所以用 ModernGPU 的 block-level sort 和 segmented reduce 按 label 聚合。
8. 最后把聚合后的 posterior 写入 grads。

*/



using namespace mgpu;

template<typename ProbT>
__device__ ProbT log_add_exp(ProbT a, ProbT b) {
    ctc_helper::log_plus<ProbT> log_plus_f;
    return log_plus_f(a, b);
}

template<typename ProbT>
__device__ bool is_major_label(int label, int major_label_offset, int num_majors) {
    return num_majors > 0 &&
           label >= major_label_offset &&
           label < major_label_offset + num_majors;
}

template<typename ProbT>
__device__ ProbT label_log_prob(const ProbT* probs,
                                int row_offset,
                                int label,
                                int out_dim,
                                const int* major_offsets,
                                const int* major_flat_subclasses,
                                int num_majors,
                                int major_label_offset) {

    /*
    给定一个 label，返回它在当前时间步的 log emission probability。
    也就是：如果 label 是普通 subclass:返回 log P(sub | x_t)
    如果 label 是 major:返回 logsumexp_{sub ∈ major} log P(sub | x_t)
    */


    if (!is_major_label<ProbT>(label, major_label_offset, num_majors)) {
        return (0 <= label && label < out_dim)
            ? probs[row_offset + label]
            : ctc_helper::neg_inf<ProbT>();
    }

    const int major = label - major_label_offset;
    const int begin = major_offsets[major];
    const int end = major_offsets[major + 1];
    /*
    从 CSR 映射中找到这个 major 包含的 subclass 范围：
    major_flat_subclasses[begin : end]

    例如：
    major_offsets = [0, 2, 5]
    major_flat_subclasses = [0, 1, 2, 3, 4]

    对于 major=1：
    begin = major_offsets[1] = 2
    end = major_offsets[2] = 5

    所以：
    major1 = {2, 3, 4}

    */

    ProbT result = ctc_helper::neg_inf<ProbT>();
    for (int i = begin; i < end; ++i) {
        const int sub = major_flat_subclasses[i];
        result = log_add_exp(result, probs[row_offset + sub]);
    }
    return result;  //返回 major 的 log emission probability
}

template<typename ProbT>
__device__ void write_label_posterior_to_grads(const ProbT* probs, // 用于将当前时间步、某个 label 的 posterior 写入 grads
                                               ProbT* grads,
                                               int row_offset,
                                               int label,
                                               ProbT state_logsum, // 当前时间步上，该 label 对应所有 CTC 状态的 alpha+beta 聚合值
                                               ProbT log_partition, // 整条标签序列的总 log-likelihood
                                               int out_dim,
                                               const int* major_offsets,
                                               const int* major_flat_subclasses,
                                               int num_majors,
                                               int major_label_offset) {
    const ProbT log_emission = label_log_prob(
        probs, row_offset, label, out_dim, major_offsets, major_flat_subclasses,
        num_majors, major_label_offset);
        //无论当前 label 是 subclass 还是 major，都先通过 label_log_prob 得到它的 emission：

    if (!is_major_label<ProbT>(label, major_label_offset, num_majors)) {
        const ProbT posterior = exp(state_logsum - log_emission - log_partition);
        atomicAdd(&grads[row_offset + label], posterior);

        /*
        这里分两种情况。
        如果 num_majors > 0，说明 latent CTC 模式启用。即使当前 label 是普通 subclass，它也可能同时收到 major label 分摊来的 posterior。
        因此使用atomicAdd把当前 posterior 累加进去。
        如果 num_majors == 0，是普通 CTC 模式，没有 major 分摊，直接赋值即可。
        */

        return;
    }

    const int major = label - major_label_offset;  // 把外部 major label 编号转成内部 major index
    const int begin = major_offsets[major]; // 取出该 major 包含的 subclass 范围
    const int end = major_offsets[major + 1];
    const ProbT major_posterior = exp(state_logsum - log_emission - log_partition); // 计算major这个集合事件在当前时间步的总posterior

    for (int i = begin; i < end; ++i) {
        const int sub = major_flat_subclasses[i];
        const ProbT responsibility =
            major_posterior * exp(probs[row_offset + sub] - log_emission);  // 把 major 的 posterior 按当前模型对 subclass 的相对概率分配给子类。
        atomicAdd(&grads[row_offset + sub], responsibility);
        /*
        把这部分责任累加到对应 subclass 的梯度 / posterior 中
        必须用 atomicAdd，因为多个线程、多个 major 状态，甚至多个 major label 都可能写入同一个 subclass
        */
    }
}

template<int NT, int VT, typename T, typename KeyT, typename Op>
struct CTASegReduce {

/*
在一个 CUDA block 内，对已经排序好的 key 做分段处理，找出每个不同 label 的起止位置。
不是最终做 reduce 的地方，而是为后面的 segmented reduction 准备：
1. 有多少个 unique label
2. 每个 unique label 在排序后数组里的起点
3. 每个 unique label 在排序后数组里的终点
4. 每个位置原来对应哪个 CTC 状态
*/


    enum {NV = NT * VT};

    union Storage {
        typename CTAScan<NT>::Storage scanStorage;
        int indices[NV];
    };

    //adapted from global kernel KernelReduceByKeyPreprocess
    __device__ static void preprocessKeys(KeyT *keys, int count,
                                          int *numUniqueLabels, int seg_start[VT],
                                          int seg_end[VT], int *scanout) {
        __shared__ Storage shared;

        const int tid = threadIdx.x;
        // Compare adjacent keys within each thread and mark discontinuities
        int endFlags = 0;
        const int first = VT * tid;
        T key = (first < count) ? keys[first] : KeyT();
        #pragma unroll
        for (int i = 0; i < VT; ++i) {
            int index = VT * tid + 1 + i;
            T next = key;
            if(index < count) {
                next = keys[index];
            }
            if(index == count || (index < count && key != next)) {
                endFlags |= 1 << i;
            }
            key = next;
        }

        __syncthreads();

        //Count the number of encountered end flags
        int scan = CTAScan<NT>::Scan(tid, popc(endFlags), shared.scanStorage, numUniqueLabels);

        __syncthreads();

        //output the unique keys
        //use indices as scratch space
        int outputPos = scan;
        #pragma unroll
        for (int i = 0; i < VT; ++i) {
        //输出 unique keys 和对应 segment 结束位置
            if ( (endFlags >> i) & 1) {
                shared.indices[outputPos] = keys[VT * tid + i];
                scanout[outputPos] = VT * tid + i;
                outputPos++;
            }
        }

        __syncthreads();

        // Create start and end
        for (int idx = tid, j = 0; idx < (*numUniqueLabels); idx += blockDim.x, ++j) {
        //生成每个 segment 的 start 和 end
            seg_start[j] = (idx == 0) ? 0 : (scanout[idx-1] + 1);
            seg_end[j] = scanout[idx];
        }

        __syncthreads();

        //copy from the scratch space back into the keys
        #pragma unroll
        for (int i = 0; i < VT; ++i) {
            keys[i * NT + tid] = shared.indices[i * NT + tid];
        }

        __syncthreads();
    }
};

// Computes forward probabilities. This fills in a T * S matrix.
// The computation starts at t=1 (2nd row) and ends at t=T-1 (last row). Each row has
// S elements where S = 2L + 1.
//
// We only need to read in probabilities corresponding to the labels, thus a sparse
// set of values are read from the probs matrix since the character set is much smaller
// than the labels. This is much more true for Mandarin than English.
template<typename ProbT, int NT, int VT>
__global__
void compute_alpha_kernel (const ProbT* probs, const int *label_sizes,
                           const int *utt_length, const int *repeats_in_labels,
                           const int *labels_without_blanks, const int *label_offsets, 
                           int *labels_with_blanks, ProbT *alphas, 
                           ProbT* ll_forward, int stride, int out_dim,
                           int S_memoffset, int T_memoffset, int blank_label,
                           const int* major_offsets,
                           const int* major_flat_subclasses,
                           int num_majors,
                           int major_label_offset) {

/*
对 batch 中每个样本计算 CTC alpha 动态规划表，并输出每个样本的 log-likelihood。
const ProbT* probs：输入 log 概率矩阵。更准确应理解为：log_probs[t, b, k] = log p(k | x_t)
const int *label_sizes：每个样本的原始标签长度 L。
const int *utt_length：每个样本的输入帧数 T。
const int *repeats_in_labels：每个样本标签中相邻重复字符个数。
const int *labels_without_blanks：原始标签，不含 blank。
const int *label_offsets：因为 batch 中标签被压平成 flat_labels，所以需要知道每个样本标签的起点。
int *labels_with_blanks：加入 blank 后的扩展标签。
ProbT *alphas：前向动态规划表。
ProbT* ll_forward：每个样本的 forward log-likelihood。
int stride：上一份代码里传入的是：stride = minibatch_，表示概率矩阵中从一个时间步跳到下一个时间步时，batch 维度的跨度。
int out_dim：输出类别数，包括 blank。
int S_memoffset、int T_memoffset：这是 batch 内统一分配时使用的最大 S 和最大 T。
    每个样本真实的：S = 2L + 1；T = utt_length[blockIdx.x]
    但是在内存中，每个样本的 alpha 表按照最大尺寸切片：
    S_memoffset × T_memoffset
int blank_label：blank 类别编号。
*/


    ctc_helper::log_plus<ProbT> log_plus_f;

    const int tid = threadIdx.x;
    const int L = label_sizes[blockIdx.x];
    const int T = utt_length[blockIdx.x];
    const int S = 2*L + 1;
    const int prob_offset = out_dim * blockIdx.x;
    const int repeats = repeats_in_labels[blockIdx.x];

    const int NV = NT * VT;
    __shared__ int label[NV];

    if ((L + repeats) > T) //判断 CTC 是否可对齐
        return;

    // Generate labels with blanks from labels without blanks
    {
        const int label_start_offset = label_offsets[blockIdx.x];
        for (int idx = tid; idx < L; idx += blockDim.x) {
            const int offset = (blockIdx.x * S_memoffset) + 2 * idx;
            labels_with_blanks[offset] = blank_label;
            labels_with_blanks[offset+1] = labels_without_blanks[label_start_offset + idx];
        }
        if (tid == 0) {
            labels_with_blanks[(blockIdx.x * S_memoffset) + 2 * L] = blank_label;
        }
    }
    __syncthreads();

    //设置当前样本的 labels 和 alpha 指针
    const int *labels = labels_with_blanks;
    const int* label_global = &labels[blockIdx.x * S_memoffset];
    ProbT* alpha = &alphas[blockIdx.x * (S_memoffset * T_memoffset)];

    // Set the first row of alpha neg_inf - it is much more efficient to do it
    // here than outside
    #pragma unroll
    for (int idx = tid; idx < min(S, NV); idx += blockDim.x) {
        alpha[idx] = ctc_helper::neg_inf<ProbT>();
    }

    // Load labels into shared memory
    //如果每次都访问 global memory，会比较慢。shared memory 更快，适合存放当前 block 内反复访问的数据。
    #pragma unroll
    for (int i = tid; i < S; i += NT) {
        label[i] = label_global[i];
    }

    __syncthreads();
    int start =  (L + repeats < T) ? 0 : 1; //说明输入帧数比最短合法路径更长，有多余时间，可以从 blank 开始。
    int end = S > 1 ? 2 : 1;

    // Initialize the first row corresponding to t=0;
    for(int i = tid; i < (end-start); i += blockDim.x)
        alpha[i + start] = label_log_prob(
            probs, prob_offset, label[i + start], out_dim,
            major_offsets, major_flat_subclasses, num_majors, major_label_offset);

    __syncthreads();

    // Fill in the rest of matrix, one row at a time (outer loop).
    for(int t = 1; t < T; ++t) {

        // Start offsets into the current and previous row
        /*计算原理：
        log_alpha(t, s)
        = logsumexp(
            log_alpha(t-1, s),
            log_alpha(t-1, s-1),
            log_alpha(t-1, s-2) 条件允许时
          )
          + log p(label_s | x_t)
        */
        const int start_cur_row = t * S;
        const int start_prev_row = (t - 1) * S;

        // The prob is a 2D column major array, with probabilites for each t strided
        // by (out_dim * stride), where stride is the minibatch size
        const int start_prob_col = t * (out_dim * stride);

        // This is the first column and in this case there is nothing left of it
        if (tid == 0) {
            if (start == 0) {
                alpha[start_cur_row] = alpha[start_prev_row] +
                                       label_log_prob(
                                           probs, prob_offset + start_prob_col,
                                           blank_label, out_dim, major_offsets,
                                           major_flat_subclasses, num_majors,
                                           major_label_offset);
            }
            else if (start == 1) {
                alpha[start_cur_row] = alpha[start_prev_row];
            }
        }

        __syncthreads();

        // Fill in the elements in each row. There is no loop dependence here since our
        // input is the row above. We sum either two or three adjacent values from the
        // row above depending on whether we have a blank or repeated characters. Finally
        // we add the probability corresponding to this label at time t
        #pragma unroll
        for (int idx = (tid+1); idx < S; idx += blockDim.x) {

            ProbT prev_sum = log_plus_f(alpha[idx + start_prev_row], alpha[(idx-1) + start_prev_row]);

            // Skip two if not on blank and not on repeat.
            if ((label[idx] != blank_label) &&
                (idx != 1) && (label[idx] != label[idx-2]))
                prev_sum = log_plus_f(prev_sum, alpha[(idx-2) + start_prev_row]);

            alpha[idx + start_cur_row] =
                prev_sum + label_log_prob(
                    probs, prob_offset + start_prob_col, label[idx], out_dim,
                    major_offsets, major_flat_subclasses, num_majors,
                    major_label_offset);
        }

        __syncthreads();
    }

    if (tid == 0) {
        // Add and return the rightmost two/one element(s) in the last row.
        ProbT loglike = ctc_helper::neg_inf<ProbT>();

        // This is the total increment for s_inc and e_inc through the loop
        const int val = 2 * (L-1) + 1 - (((L + repeats) == T) ? 1 : 0);

        start = (val * (L!=0) + start);
        end = (val * (L!=0) + end);

        for(int i = start; i < end; ++i)
            loglike = log_plus_f(loglike, alpha[i + (T - 1) * S]);

        ll_forward[blockIdx.x] = loglike; //P(y|x) = alpha(T-1, S-1) + alpha(T-1, S-2)
    }
}

// Computes backward probabilities. This also fills in a T * S matrix
//
// See comments above compute_alphas for more context.
template<typename ProbT, int NT, int VT>
__global__
void compute_betas_and_grad_kernel (const ProbT* probs, const int *label_sizes,
                                    const int *utt_length, const int *repeats_in_labels,
                                    const int *labels_with_blanks, ProbT *alphas,
                                    const ProbT* ll_forward, ProbT *ll_backward,
                                    ProbT *grads, int stride, int out_dim,
                                    int S_memoffset, int T_memoffset, int blank_label,
                                    const int* major_offsets,
                                    const int* major_flat_subclasses,
                                    int num_majors,
                                    int major_label_offset) {

/*
1. 计算 beta 后向概率
2. 结合 alpha 和 beta 得到每个状态的 posterior
3. 把同一 label 的多个状态聚合
4. 写出 grads
*/
    ctc_helper::log_plus<ProbT> log_plus_f;
    typedef CTASegReduce<NT, VT, ProbT, int, ctc_helper::log_plus<ProbT>> SegReduce;

    const int tid = threadIdx.x;
    const int L = label_sizes[blockIdx.x];
    const int T = utt_length[blockIdx.x];
    const int S = 2*L + 1;
    const int prob_offset = out_dim * blockIdx.x;
    const int repeats = repeats_in_labels[blockIdx.x];
    const ProbT log_partition = ll_forward[blockIdx.x]; //log P(y | x)

    const int* labels = labels_with_blanks;
    const int* label_global = &labels[blockIdx.x * S_memoffset];
    ProbT* alpha = &alphas[blockIdx.x * (S_memoffset * T_memoffset)];

    const int NV = NT * VT;

    union TempStorage {
        ProbT beta[NV];
        int result[NV];
    };

    __shared__ TempStorage temp_buffer;

    __shared__ int label[NV];

    // Temporaries needed for segmented reduce
    // TODO: see if we can combine the shared memory requirements
    __shared__ int keys_shared[NV];
    __shared__ int gather_indices[NV];
    __shared__ ProbT output[NV];
    /*
    keys_shared：排序后的 label key。
    gather_indices：排序后位置对应原始 CTC 状态下标。
    output：当前时间步每个 CTC 状态的 alpha + beta。
    */
    ProbT beta_val[VT];

    if ((L + repeats) > T)
        return;

    int start = S > 1 ? (S - 2) : 0;
    int end = (L + repeats < T) ? S : S-1;

    // Setup shared memory buffers
    #pragma unroll
    for (int idx = tid; idx < NV; idx += NT) {
        label[idx] = (idx < S) ? label_global[idx] : INT_MAX;
    }

    __syncthreads();

    // int flags;
    int uniquelabels = 0;
    int seg_start[VT];
    int seg_end[VT];

    if (false) {
        // Sort labels and record indices from which to gather from.
        // The legacy ModernGPU path is fragile on some Hopper/CUDA12 builds.
        // Gradients are accumulated directly with atomicAdd below instead.
        {
            int key[VT];
            int gather_val[VT];

            #pragma unroll
            for (int i = 0; i < VT; ++i) {
                const int idx = tid * VT + i;
                gather_val[i] = idx;
                key[i] = label[idx];
            }

            __syncthreads();

            CTAMergesort<NT, VT, true, true, int, int, mgpu::less<int>>
                (key, gather_val, keys_shared, gather_indices, S, tid, mgpu::less<int>());

            __syncthreads();

            for (int i = 0; i < VT; ++i) {
                const int idx = tid * VT + i;
                gather_indices[idx] = gather_val[i];
            }

            __syncthreads();

            SegReduce::preprocessKeys(keys_shared, S, &uniquelabels, seg_start, seg_end,
                                      temp_buffer.result);
            __syncthreads();
        }

        // TODO: probably not necessary
        __syncthreads();
    }

    // Load labels back
    #pragma unroll
    for (int idx = tid; idx < NV; idx += NT) {
        temp_buffer.beta[idx] = ctc_helper::neg_inf<ProbT>();
    }
    __syncthreads();

    // Initialize the two rightmost values in the last row (assuming L non-zero)
    for(int i = tid; i < (end-start); i += blockDim.x)
        temp_buffer.beta[i + start] =
            label_log_prob(
                probs, prob_offset + (T - 1) * (out_dim * stride),
                label[i + start], out_dim, major_offsets, major_flat_subclasses,
                num_majors, major_label_offset);

    __syncthreads();

    // Load output data in registers through the transpose trick - should really be a function
    //初始化最后一帧 output
    #pragma unroll
    for (int idx = tid; idx < S; idx += NT) {
        output[idx] = alpha[idx + (T - 1) * S] + temp_buffer.beta[idx];
    }

    __syncthreads();

    // Start at the second to last row and backward in time
    for(int t = T - 1; t >= 0; --t) {

        // Start offsets into the current and next row
        const int start_cur_row = t * S;

        // Starting offset of column that we read from the probs array
        const int start_prob_col = t * (out_dim * stride);

        if (t < T-1) {

            // Filling up one row at at time but going back in time from the last row
            // to the first. As in the forward pass, there is no loop dependence and we
            // do a variable length filter of maximum filter size of 3
            #pragma unroll
            for(int idx = tid, i = 0; idx < (S-1); idx += NT, i++) {
                ProbT next_sum = log_plus_f(temp_buffer.beta[idx], temp_buffer.beta[idx+1]);

                    // Skip two if not on blank and not on repeat.
                if ((label[idx] != blank_label) &&
                    (idx != (S-2)) && (label[idx] != label[idx+2]))
                    next_sum = log_plus_f(next_sum, temp_buffer.beta[idx+2]);

                beta_val[i] = next_sum + label_log_prob(
                    probs, prob_offset + start_prob_col, label[idx], out_dim,
                    major_offsets, major_flat_subclasses, num_majors,
                    major_label_offset);
            }

            __syncthreads();

            // Initialize values for the rightmost column since there is nothing to the right
            // Update input buffer for next iteration
            if ((tid == 0) && (end == S))
                temp_buffer.beta[(S-1)] = temp_buffer.beta[(S-1)] +
                                          label_log_prob(
                                              probs, prob_offset + start_prob_col,
                                              blank_label, out_dim, major_offsets,
                                              major_flat_subclasses, num_majors,
                                              major_label_offset);

            #pragma unroll
            for(int idx = tid, i = 0; idx < (S-1); idx += NT, i++) {
               temp_buffer.beta[idx] = beta_val[i];
            }

            __syncthreads();

            // Beta Computation done - add to alpha and update the gradient. Reload
            // the gradient back for segmented reduce later on
            #pragma unroll
            for(int idx = tid; idx < S; idx += NT) {
               output[idx] = alpha[idx + start_cur_row] + temp_buffer.beta[idx];
            }

            __syncthreads();

        }

        __syncthreads();

        // Compute posterior contribution for each CTC state.
        //
        // In latent mode, write_label_posterior_to_grads() uses atomicAdd for
        // both subclass labels and major labels.  Therefore we can accumulate
        // directly over states and avoid the ModernGPU block sort/seg-reduce
        // path.  That path is fragile on some Hopper/CUDA12 builds for small
        // S in a mixed-length minibatch (for example label lengths [180, 55]).
        if (true) {
            for (int idx = tid; idx < S; idx += blockDim.x) {
                write_label_posterior_to_grads(
                    probs, grads, prob_offset + start_prob_col, label[idx],
                    output[idx], log_partition, out_dim, major_offsets,
                    major_flat_subclasses, num_majors, major_label_offset);
            }
            __syncthreads();
        }
        else {
        // Compute segmented reduction of output by using label as key
        {
            // Somewhat faster key value reduce
            ProbT accum[VT];

            for (int idx = tid, j = 0; idx < uniquelabels; idx += blockDim.x, ++j) {

                accum[j] = ctc_helper::neg_inf<ProbT>();
                for (int i = seg_start[j]; i <= seg_end[j]; ++i) {
                    accum[j] = log_plus_f(accum[j], output[gather_indices[i]]);
                }
            }
            __syncthreads();

            // Write accumulated value into output since that is not used
            // for (int idx = tid, j = 0; idx < uniquelabels; idx += blockDim.x, ++j) {
            //     output[idx] = accum[j];
            // }
            // __syncthreads();

            // for (int idx = tid; idx < out_dim; idx += blockDim.x) {
            //     const int grads_offset = prob_offset + start_prob_col + idx;
            //     grads[grads_offset] = probs[grads_offset];
            // }

            // __syncthreads();

            // for (int idx = tid; idx < uniquelabels; idx += blockDim.x) {
            //     const int grads_offset = prob_offset + start_prob_col + keys_shared[idx];

            //     ProbT grad = output[idx];

            //     if ((grad == 0.0) || (probs[grads_offset] == 0.0) ||
            //         (grad == ctc_helper::neg_inf<ProbT>())) {
            //     } else {
            //         grads[grads_offset] =
            //             probs[grads_offset] - exp(grad - log(probs[grads_offset]) - log_partition);
            //     }
            // }

            // grad for softmax
            for (int idx = tid, j = 0; idx < uniquelabels; idx += blockDim.x, ++j) {
                 write_label_posterior_to_grads(
                     probs, grads, prob_offset + start_prob_col, keys_shared[idx],
                     accum[j], log_partition, out_dim, major_offsets,
                     major_flat_subclasses, num_majors, major_label_offset);

                 /*
                相加后，当前帧概率被算了两次。
                真正的状态后验应当类似：
                posterior(t, s)
                = exp(alpha(t,s) + beta(t,s)
                      - log p(label_s | x_t)
                      - log P(y|x))
                最后得到当前时间步 t 上，类别 k 对目标序列 y 的 CTC 后验责任
                 */
            }
            __syncthreads();
        }
        }

        // Output backward log likelihood
        //backward 是在第一帧的左端状态求和。
        if ((t == 0) && (tid == 0)) {
            ProbT loglike = ctc_helper::neg_inf<ProbT>();

            const int val = 2 * (L-1) + 1 - (((L + repeats) == T) ? 1 : 0);

            start = (-val * (L != 0) + start);
            end = (-val * (L != 0) + end);

            // Sum and return the leftmost one/two value(s) in first row
            for(int i = start; i < end; ++i)
                loglike = log_plus_f(loglike, temp_buffer.beta[i]);

            ll_backward[blockIdx.x] = loglike;
        }

        // For some reason this is important
        __syncthreads();
    }
}
