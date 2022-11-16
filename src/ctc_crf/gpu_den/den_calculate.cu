// Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang
// Apache 2.0.
// This file contains functions for calculating the denominator gradients in log domain.

#include <cstdio>
#include <cstdlib>
#include <vector>
// for each state
// start_weight
// end_weight

// Transition: float weight, int input_label, int state
// alpha_transition_index 
// beta_transition_index

#define CHECK_CUDA(call) \
{  \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}
#define ATOMIC_CONST 32
#define CU_BLOCK_DIM 1024

__host__ __device__
inline float log_plus(float a, float b) {
    if (a == -float(INFINITY)) return b;
    if (b == -float(INFINITY)) return a;
    float m = a > b ? a : b;
    return log1pf(expf(-fabs(a - b))) + m;
}

__device__ float atomic_log_plus(float *addr_f, float value) {
    int *addr = (int*)addr_f;
    float expected = *addr_f;
    float sum = log_plus(expected, value);
    int old_value = atomicCAS(addr, __float_as_int(expected), __float_as_int(sum));

    while (old_value != __float_as_int(expected)) {
        expected = __int_as_float(old_value);
        sum = log_plus(expected, value);
        old_value = atomicCAS(addr, __float_as_int(expected), __float_as_int(sum));
    }
    return __int_as_float(old_value);
}

struct Transition {
    float weight = -float(INFINITY);
    int label = 0;
    int state = 0;
};

struct IntPair {
    int first = 1;
    int second = 0;
};

// <<<batch_size, CU_BLOCK_CONST>>>
__global__ void alpha_first_kernel(float *alpha,
                                   const int alpha_size,
                                   const int batch_size,
                                   const int T,
                                   const float * const start_weight) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    for (int idx = tid; idx < alpha_size; idx += blockDim.x) {
        alpha[mini_batch_idx * alpha_size * (T+1) + idx] = start_weight[idx];
    }
}

__global__ void alpha_kernel(float *alpha,
                             const float* const logits,
                             const int batch_size,
                             const int T,
                             const int t,
                             const int * const input_lengths,
                             const int alpha_size,
                             const int logits_size,
                             const IntPair * const alpha_transition_index,
                             const Transition * const alpha_transition) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (t > input_lengths[mini_batch_idx]) return;

    int idx1 = mini_batch_idx * alpha_size * (T+1) + alpha_size * t;
    int idx2 = mini_batch_idx * alpha_size * (T+1) + alpha_size * (t-1);
    int idx3 = mini_batch_idx * logits_size * T + logits_size * (t-1);

    for (int idx = tid; idx < alpha_size; idx += blockDim.x) {
        int start = alpha_transition_index[idx].first;
        int end = alpha_transition_index[idx].second;
        float result = -float(INFINITY);
        for (int k = start; k <= end; k++) {
            result = log_plus(alpha[idx2+alpha_transition[k].state] + 
                alpha_transition[k].weight + logits[idx3+alpha_transition[k].label], result);
        }
        alpha[idx1+idx] = result;
    }
}

__global__ void alpha_last_kernel(float *alpha,
                                  const int alpha_size,
                                  const int batch_size,
                                  const int T,
                                  const int * const input_lengths,
                                  const float * const end_weight) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int alpha_start = mini_batch_idx * alpha_size * (T+1);
    int cT = input_lengths[mini_batch_idx];

    for (int idx = tid; idx < alpha_size; idx += blockDim.x) {
        alpha[alpha_start+cT*alpha_size+idx] += end_weight[idx];
    }
}

// <<< minibatch, N = 32,64,128...>>>
__global__ void alpha_lld_kernal(const float * const alpha,
                                 const int alpha_size,
                                 const int T,
                                 const int * const input_lengths,
                                 float * loglikelihood) {
    int mini_batch_idx = blockIdx.x;
    int idx = threadIdx.x;
    int block_dim = blockDim.x;
    int cT = input_lengths[mini_batch_idx];
    int last_idx = alpha_size * (T+1) * mini_batch_idx + cT*alpha_size;
    // printf("enter alpha_lld_kernal, block.x: %d, thread.x: %d\n", blockIdx.x, threadIdx.x);

    extern __shared__ float sdata[];
    float temp = -float(INFINITY);

    for (int i = idx; i < alpha_size; i += block_dim) {
        temp = log_plus(temp, alpha[last_idx+i]);
    }
    sdata[idx] = temp;
    __syncthreads();

    for (int shift = block_dim / 2; shift > warpSize; shift >>= 1) {
        if (idx < shift) {
            sdata[idx] = log_plus(sdata[idx], sdata[idx+shift]);
        }
        __syncthreads();
    }

    if (idx < warpSize) {
        for (int shift = warpSize; shift > 0; shift >>= 1) {
            sdata[idx] = log_plus(sdata[idx], sdata[idx+shift]);
        }
    }
    __syncthreads();

    if (idx == 0) {
        loglikelihood[mini_batch_idx] = sdata[0];
        // printf("alpha loglikelihod: %f mini_batch %d\n", loglikelihood[mini_batch_idx], mini_batch_idx);
    }
}

__global__ void beta_last_kernel(float *beta,
                                 const int beta_size,
                                 const int batch_size,
                                 const int * const input_lengths,
                                 const float * const end_weight) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int cT = input_lengths[mini_batch_idx];

    for (int idx = tid; idx < beta_size; idx += blockDim.x) {
        beta[mini_batch_idx * 2 * beta_size + (cT % 2) * beta_size + idx] = end_weight[idx];
    }
}

__global__ void beta_first_kernel(float *beta, 
                                  const int beta_size,
                                  const int batch_size,
                                  const float * const start_weight) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    for (int idx = tid; idx < beta_size; idx += blockDim.x) {
        beta[mini_batch_idx * 2 * beta_size + idx] += start_weight[idx];
    }
}

__global__ void beta_kernel(float *beta,
                            const float* const alpha,
                            const float* const logits, 
                            float *grad_storage,
                            const int batch_size,
                            const int T,
                            const int t,
                            const int *input_lengths,
                            const int beta_size,
                            const int logits_size,
                            const IntPair * const beta_transition_index,
                            const Transition * const beta_transition) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= input_lengths[mini_batch_idx]) return;
    int idx1 = mini_batch_idx * beta_size * (T+1) + beta_size * t;
    int idx2 = mini_batch_idx * beta_size * 2 + beta_size * ((t+1) % 2);
    int idx3 = mini_batch_idx * beta_size * 2 + beta_size * (t % 2);
    int idx4 = mini_batch_idx * logits_size * T + logits_size * t;
    int idx5 = mini_batch_idx * logits_size * ATOMIC_CONST;

    for (int idx = tid; idx < beta_size; idx += blockDim.x) {
        int start = beta_transition_index[idx].first;
        int end = beta_transition_index[idx].second;

        float beta_result = -float(INFINITY);
        float temp_value = -float(INFINITY);

        for (int k = start; k <= end; k++) {
            temp_value = beta[idx2+beta_transition[k].state] + beta_transition[k].weight +
                logits[idx4+beta_transition[k].label];
            beta_result = log_plus(temp_value, beta_result);
            float partial_grad = alpha[idx1+idx] + temp_value; 
            float *grad_position = grad_storage + idx5 + beta_transition[k].label * ATOMIC_CONST + threadIdx.x % ATOMIC_CONST;
            atomic_log_plus(grad_position, partial_grad);
        }
        beta[idx3+idx] = beta_result;
    }
}

__global__ void copy_grad(float *grad_storage,
                          float *grad_net,
                          const float * const alpha_lld,
                          const int * const input_lengths,
                          const int batch_size,
                          const int logits_size,
                          const int T,
                          const int t) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= input_lengths[mini_batch_idx]) return;

    float lld = alpha_lld[mini_batch_idx];
    for (int idx = tid; idx < logits_size; idx += blockDim.x) {
        float *grad_position = grad_net + mini_batch_idx*logits_size*T + t*logits_size + idx;
        int idx_storage = mini_batch_idx*logits_size*ATOMIC_CONST+idx*ATOMIC_CONST;

        float grad = -float(INFINITY);
        for (int i = 0; i < ATOMIC_CONST; i++) {
            grad = log_plus(grad_storage[idx_storage+i], grad);
            grad_storage[idx_storage+i] = -float(INFINITY);
        }
        *grad_position = expf(grad - lld);
    }
}

__global__ void beta_lld_kernal(const float * const beta,
                                const int beta_size,
                                float * loglikelihood) {
    int idx = threadIdx.x;
    int first_idx = beta_size * 2 * idx;
    loglikelihood[idx] = beta[first_idx];
}

Transition ** TRANSITION_ALPHA = NULL;
Transition ** TRANSITION_BETA = NULL;
IntPair ** TRANSITION_INDEX_ALPHA = NULL;
IntPair ** TRANSITION_INDEX_BETA = NULL;
float ** START_WEIGHT = NULL;
float ** END_WEIGHT = NULL;

int DEN_NUM_ARCS = 0;
int DEN_NUM_STATES = 0;

int *DEVICE_HASH = NULL;

void ReadFst(const char * fst_name,
             std::vector<std::vector<int> > &alpha_next,
             std::vector<std::vector<int> > &beta_next,
             std::vector<std::vector<int> > &alpha_ilabel,
             std::vector<std::vector<int> > &beta_ilabel,
             std::vector<std::vector<float> > &alpha_weight,
             std::vector<std::vector<float> > &beta_weight,
             std::vector<float> &start_weight,
             std::vector<float> &end_weight,
             int &num_states,
             int &num_arcs);

extern "C" {
void Init(const char * fst_name, int n_gpus, int * gpus) {
    std::vector<std::vector<int> > alpha_next;
    std::vector<std::vector<int> > beta_next;
    std::vector<std::vector<int> > alpha_ilabel;
    std::vector<std::vector<int> > beta_ilabel;
    std::vector<std::vector<float> > alpha_weight;
    std::vector<std::vector<float> > beta_weight;
    std::vector<float> start_weight;
    std::vector<float> end_weight;

    // const char * fst_name = "test_lm.fst";
    int num_states = 0;
    int num_arcs = 0;

    ReadFst(fst_name, alpha_next, beta_next, alpha_ilabel, beta_ilabel, 
        alpha_weight, beta_weight, start_weight, end_weight, num_states, num_arcs);
    
    DEN_NUM_ARCS = num_arcs;
    DEN_NUM_STATES = num_states;


    std::vector<Transition> transition_alpha(num_arcs);
    std::vector<Transition> transition_beta(num_arcs);
    std::vector<IntPair> transition_index_alpha(num_states);
    std::vector<IntPair> transition_index_beta(num_states);


    int count = 0;
    for (int i = 0; i < num_states; i++) {
        if (alpha_next[i].empty()) {
            transition_index_alpha[i].first = 1;
            transition_index_alpha[i].second = 0;
        } else {
            transition_index_alpha[i].first = count;
            for (int j = 0; j < alpha_next[i].size(); j++) {
                transition_alpha[count].state = alpha_next[i][j];
                transition_alpha[count].label = alpha_ilabel[i][j];
                transition_alpha[count].weight = alpha_weight[i][j];
                count++;
            }
            transition_index_alpha[i].second = count-1;
        }
    }
    if (count != num_arcs) {
        fprintf(stderr, "count does not equal to num_arcs\n");
        exit(-1);
    }

    count = 0;
    for (int i = 0; i < num_states; i++) {
        if (beta_next[i].empty()) {
            transition_index_beta[i].first = 1;
            transition_index_beta[i].second = 0;
        } else {
            transition_index_beta[i].first = count;
            for (int j = 0; j < beta_next[i].size(); j++) {
                transition_beta[count].state = beta_next[i][j];
                transition_beta[count].label = beta_ilabel[i][j];
                transition_beta[count].weight = beta_weight[i][j];
                count++;
            }
            transition_index_beta[i].second = count-1;
        }
    }
    if (count != num_arcs) {
        fprintf(stderr, "count does not equal to num_arcs\n");
        exit(-1);
    }

    int max_gpu = 0;
    for (int i = 0; i < n_gpus; i++) {
        if (gpus[i] > max_gpu) max_gpu = gpus[i];
    }
    DEVICE_HASH = new int[max_gpu+1];
    memset(DEVICE_HASH, 0, sizeof(int)*(max_gpu+1));
    for (int i = 0; i < n_gpus; i++) DEVICE_HASH[gpus[i]] = i;

    TRANSITION_ALPHA = new Transition*[n_gpus];
    TRANSITION_BETA= new Transition*[n_gpus];
    TRANSITION_INDEX_ALPHA = new IntPair*[n_gpus];
    TRANSITION_INDEX_BETA= new IntPair*[n_gpus];
    START_WEIGHT = new float*[n_gpus];
    END_WEIGHT = new float*[n_gpus];

    int prev_device = 0;
    CHECK_CUDA(cudaGetDevice(&prev_device));

    for (int i = 0; i < n_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(gpus[i]));
        CHECK_CUDA(cudaMalloc((void**)&TRANSITION_ALPHA[i], sizeof(Transition)*num_arcs));
        CHECK_CUDA(cudaMalloc((void**)&TRANSITION_BETA[i], sizeof(Transition)*num_arcs));
        CHECK_CUDA(cudaMalloc((void**)&TRANSITION_INDEX_ALPHA[i], sizeof(IntPair)*num_states));
        CHECK_CUDA(cudaMalloc((void**)&TRANSITION_INDEX_BETA[i], sizeof(IntPair)*num_states));
        CHECK_CUDA(cudaMalloc((void**)&START_WEIGHT[i], sizeof(float)*num_states));
        CHECK_CUDA(cudaMalloc((void**)&END_WEIGHT[i], sizeof(float)*num_states));

        CHECK_CUDA(cudaMemcpy(TRANSITION_ALPHA[i], transition_alpha.data(), sizeof(Transition)*num_arcs, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(TRANSITION_BETA[i], transition_beta.data(), sizeof(Transition)*num_arcs, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(TRANSITION_INDEX_ALPHA[i], transition_index_alpha.data(), sizeof(IntPair)*num_states, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(TRANSITION_INDEX_BETA[i], transition_index_beta.data(), sizeof(IntPair)*num_states, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(START_WEIGHT[i], start_weight.data(), sizeof(float)*num_states, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(END_WEIGHT[i], end_weight.data(), sizeof(float)*num_states, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaSetDevice(prev_device));
}

void Release(int n_gpus, int *gpus) {
    int prev_device = 0;
    CHECK_CUDA(cudaGetDevice(&prev_device));

    for (int i = 0; i < n_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(gpus[i]));

        CHECK_CUDA(cudaFree(TRANSITION_ALPHA[i]));
        CHECK_CUDA(cudaFree(TRANSITION_BETA[i]));
        CHECK_CUDA(cudaFree(TRANSITION_INDEX_ALPHA[i]));
        CHECK_CUDA(cudaFree(TRANSITION_INDEX_BETA[i]));
        CHECK_CUDA(cudaFree(START_WEIGHT[i]));
        CHECK_CUDA(cudaFree(END_WEIGHT[i]));
    }
    CHECK_CUDA(cudaSetDevice(prev_device));
    delete[] TRANSITION_ALPHA;
    delete[] TRANSITION_BETA;
    delete[] TRANSITION_INDEX_ALPHA;
    delete[] TRANSITION_INDEX_BETA;
    delete[] START_WEIGHT;
    delete[] END_WEIGHT;

    TRANSITION_ALPHA = NULL;
    TRANSITION_BETA = NULL;
    TRANSITION_INDEX_ALPHA = NULL;
    TRANSITION_INDEX_BETA = NULL;
    START_WEIGHT = NULL;
    END_WEIGHT = NULL;

    delete[] DEVICE_HASH;
    DEVICE_HASH = NULL;
}

void compute_alpha(float *alpha,
                   float *logits,
                   const int batch_size,
                   int T,
                   const int alpha_size,
                   int logits_size,
                   int *input_lengths,
                   float * loglikelihood,
                   cudaStream_t stream) {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    int gid = DEVICE_HASH[device];

    int alpha_lld_dim = 128;
    alpha_first_kernel<<<batch_size, CU_BLOCK_DIM, 0, stream>>>(alpha, alpha_size, batch_size, T, START_WEIGHT[gid]);

    for (int t = 1; t <= T; t++) {
        alpha_kernel<<<batch_size, CU_BLOCK_DIM, 0, stream>>>(alpha, logits, batch_size, T, t, input_lengths, 
            alpha_size, logits_size, TRANSITION_INDEX_ALPHA[gid], TRANSITION_ALPHA[gid]);
    }

    alpha_last_kernel<<<batch_size, CU_BLOCK_DIM, 0, stream>>>(alpha, alpha_size, batch_size, T, input_lengths, END_WEIGHT[gid]);
    alpha_lld_kernal<<<batch_size, alpha_lld_dim, sizeof(float)*alpha_lld_dim, stream>>>(alpha, alpha_size, T, input_lengths, loglikelihood);
    // cudaDeviceSynchronize();
}

void compute_beta_and_grad(float *beta,
                           const float * const alpha,
                           const float * const logits,
                           const float * const alpha_lld,
                           float *grad_storage,
                           float *grad_net,
                           const int batch_size,
                           const int T,
                           const int beta_size,
                           const int logits_size,
                           const int * const input_lengths,
                           float * loglikelihood,
                           cudaStream_t stream) {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    int gid= DEVICE_HASH[device];
    // set grad_storage
    copy_grad<<<batch_size, CU_BLOCK_DIM, 0, stream>>>(grad_storage, grad_net, alpha_lld, input_lengths, batch_size, logits_size, T, 0);

    beta_last_kernel<<<batch_size, CU_BLOCK_DIM, 0, stream>>>(beta, beta_size, batch_size, input_lengths, END_WEIGHT[gid]);
    for (int t = T-1; t >= 0; t--) {
        beta_kernel<<<batch_size, CU_BLOCK_DIM, 0, stream>>>(beta, alpha, logits, grad_storage, batch_size, T, t, input_lengths, beta_size, logits_size,
            TRANSITION_INDEX_BETA[gid], TRANSITION_BETA[gid]);
        copy_grad<<<batch_size, CU_BLOCK_DIM, 0, stream>>>(grad_storage, grad_net, alpha_lld, input_lengths, batch_size, logits_size, T, t);
    }

    beta_first_kernel<<<batch_size, CU_BLOCK_DIM, 0, stream>>>(beta, beta_size, batch_size, START_WEIGHT[gid]);
    beta_lld_kernal<<<1, batch_size, 0, stream>>>(beta, beta_size, loglikelihood);
}
}
