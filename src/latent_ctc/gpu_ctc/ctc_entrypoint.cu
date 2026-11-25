#include <cstddef>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include "ctc.h"
#include "gpu_ctc.h"

namespace {

inline size_t align_bytes(size_t offset, size_t alignment = 256) {
    return (offset + alignment - 1) / alignment * alignment;
}

ctcStatus_t base_get_workspace_size_impl(const int* const label_lengths,
                                         const int* const input_lengths,
                                         int alphabet_size,
                                         int minibatch,
                                         size_t* size_bytes) {
    if (label_lengths == nullptr ||
        input_lengths == nullptr ||
        size_bytes == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0) {
        return CTC_STATUS_INVALID_VALUE;
    }

    int maxL = *std::max_element(label_lengths, label_lengths + minibatch);
    int maxT = *std::max_element(input_lengths, input_lengths + minibatch);
    const int S = 2 * maxL + 1;

    *size_bytes = 0;
    *size_bytes += 2 * sizeof(float) * minibatch;          // nll_forward, nll_backward
    *size_bytes += sizeof(int) * minibatch;                // repeats
    *size_bytes += sizeof(int) * minibatch;                // label offsets
    *size_bytes += sizeof(int) * minibatch;                // utt_length
    *size_bytes += sizeof(int) * minibatch;                // label lengths
    *size_bytes += sizeof(int) * maxL * minibatch;         // labels without blanks
    *size_bytes += sizeof(int) * S * minibatch;            // labels with blanks
    *size_bytes += sizeof(float) * S * maxT * minibatch;   // alphas

    return CTC_STATUS_SUCCESS;
}

ctcStatus_t validate_latent_inputs(const int* const flat_labels,
                                   const int* const label_lengths,
                                   const int* const input_lengths,
                                   int alphabet_size,
                                   int minibatch,
                                   const int* const major_offsets,
                                   const int* const major_flat_subclasses,
                                   int num_majors,
                                   int major_label_offset,
                                   int* total_label_count,
                                   int* maxT) {
    /*
    用于检查：major 映射表是否合法；major label 编号是否和 subclass label 冲突；flat_labels 中每个标签是否合法。同时统计 total_label_count 和 maxT
    flat_labels：压平后的标签序列，允许包含 subclass label 和 major label
    major_offsets / major_flat_subclasses：CSR 风格 major->subclass 映射
    num_majors：major 类别数
    major_label_offset：major 标签编号起点
    total_label_count：输出，总标签数
    maxT：输出，batch 中最大输入长度
    */


    if (major_offsets == nullptr || major_flat_subclasses == nullptr) {
        return CTC_STATUS_INVALID_VALUE;
    }

    if (major_label_offset < alphabet_size) {
        return CTC_STATUS_INVALID_VALUE;
    }

    const int total_sub_links = major_offsets[num_majors];
    if (total_sub_links <= 0) {
        return CTC_STATUS_INVALID_VALUE;
    }

    for (int m = 0; m < num_majors; ++m) {
        const int begin = major_offsets[m];
        const int end = major_offsets[m + 1];
        if (begin < 0 || end <= begin || end > total_sub_links) {
            return CTC_STATUS_INVALID_VALUE;
        }

        for (int i = begin; i < end; ++i) {
            const int sub = major_flat_subclasses[i];
            if (sub < 0 || sub >= alphabet_size) {
                return CTC_STATUS_INVALID_VALUE;
            }
        }
    }

    *total_label_count = 0;
    *maxT = 0;
    for (int b = 0; b < minibatch; ++b) {
        if (label_lengths[b] < 0 || input_lengths[b] < 0) {
            return CTC_STATUS_INVALID_VALUE;
        }
        *total_label_count += label_lengths[b];
        *maxT = std::max(*maxT, input_lengths[b]);
    }

    for (int i = 0; i < *total_label_count; ++i) {
        const int label = flat_labels[i];
        if (0 <= label && label < alphabet_size) {
            continue;
        }

        const int major = label - major_label_offset;
        if (0 <= major && major < num_majors) {
            continue;
        }

        return CTC_STATUS_INVALID_VALUE;
    }

    return CTC_STATUS_SUCCESS;
}

} // namespace

extern "C" {

const char* ctcGetStatusString(ctcStatus_t status) {
    switch (status) {
    case CTC_STATUS_SUCCESS:
        return "no error";
    case CTC_STATUS_MEMOPS_FAILED:
        return "cuda memcpy or memset failed";
    case CTC_STATUS_INVALID_VALUE:
        return "invalid value";
    case CTC_STATUS_EXECUTION_FAILED:
        return "execution failed";
    case CTC_STATUS_UNKNOWN_ERROR:
    default:
        return "unknown error";
    }
}

ctcStatus_t compute_ctc_loss(const float* const activations,
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float *costs,
                             void *workspace,
                             ctcOptions options) {
    if (activations == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        workspace == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0) {
        return CTC_STATUS_INVALID_VALUE;
    }

    if (gradients != nullptr) {
        const int maxT = *std::max_element(input_lengths, input_lengths + minibatch);
        cudaError_t cuda_status = cudaMemsetAsync(
            gradients, 0,
            static_cast<size_t>(maxT) * minibatch * alphabet_size * sizeof(float),
            options.stream);
        if (cuda_status != cudaSuccess) {
            return CTC_STATUS_MEMOPS_FAILED;
        }
    }

    GpuCTC<float> ctc(alphabet_size, minibatch, workspace, options.stream,
                      options.blank_label);

    if (gradients != nullptr)
        return ctc.cost_and_grad(activations, gradients, costs,
                                 flat_labels, label_lengths, input_lengths);

    return ctc.score_forward(activations, costs,
                             flat_labels, label_lengths, input_lengths);
}

ctcStatus_t get_workspace_size(const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size,
                               int minibatch,
                               ctcOptions options,
                               size_t* size_bytes) {
    (void)options;
    return base_get_workspace_size_impl(label_lengths, input_lengths,
                                        alphabet_size, minibatch, size_bytes);
}

ctcStatus_t get_latent_workspace_size(const int* const label_lengths,
                                      const int* const input_lengths,
                                      int alphabet_size,
                                      int minibatch,
                                      ctcOptions options,
                                      const int* const major_offsets,
                                      int num_majors,
                                      size_t* size_bytes) {
    (void)options;
    if (label_lengths == nullptr ||
        input_lengths == nullptr ||
        size_bytes == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0 ||
        num_majors < 0) {
        return CTC_STATUS_INVALID_VALUE;
    }

    if (num_majors == 0) {
        return base_get_workspace_size_impl(label_lengths, input_lengths,
                                            alphabet_size, minibatch, size_bytes);
    }

    if (major_offsets == nullptr || major_offsets[num_majors] <= 0) {
        return CTC_STATUS_INVALID_VALUE;
    }

    size_t base_workspace = 0;
    ctcStatus_t status = base_get_workspace_size_impl(
        label_lengths, input_lengths, alphabet_size, minibatch, &base_workspace);
    if (status != CTC_STATUS_SUCCESS) {
        return status;
    }

    size_t bytes = align_bytes(base_workspace);
    bytes += static_cast<size_t>(num_majors + 1) * sizeof(int);
    bytes = align_bytes(bytes);
    bytes += static_cast<size_t>(major_offsets[num_majors]) * sizeof(int);

    *size_bytes = bytes;
    return CTC_STATUS_SUCCESS;
}

ctcStatus_t compute_latent_ctc_loss(const float* const activations,
                                    float* gradients,
                                    const int* const flat_labels,
                                    const int* const label_lengths,
                                    const int* const input_lengths,
                                    int alphabet_size,
                                    int minibatch,
                                    float *costs,
                                    void *workspace,
                                    ctcOptions options,
                                    const int* const major_offsets,
                                    const int* const major_flat_subclasses,
                                    int num_majors,
                                    int major_label_offset) {
    if (activations == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        workspace == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0 ||
        num_majors < 0) {
        return CTC_STATUS_INVALID_VALUE;
    }

    if (num_majors == 0) {
        return compute_ctc_loss(activations, gradients, flat_labels,
                                label_lengths, input_lengths, alphabet_size,
                                minibatch, costs, workspace, options);
    }

    int total_label_count = 0;
    int maxT = 0;
    ctcStatus_t status = validate_latent_inputs(
        flat_labels, label_lengths, input_lengths, alphabet_size, minibatch,
        major_offsets, major_flat_subclasses, num_majors, major_label_offset,
        &total_label_count, &maxT);
    if (status != CTC_STATUS_SUCCESS) {
        return status;
    }

    bool has_major_label = false;
    for (int i = 0; i < total_label_count; ++i) {
        const int major = flat_labels[i] - major_label_offset;
        if (0 <= major && major < num_majors) {
            has_major_label = true;
            break;
        }
    }

    if (!has_major_label) {
        return compute_ctc_loss(activations, gradients, flat_labels,
                                label_lengths, input_lengths, alphabet_size,
                                minibatch, costs, workspace, options);
    }

    size_t inner_workspace_size = 0;
    status = base_get_workspace_size_impl(label_lengths, input_lengths,
                                          alphabet_size, minibatch,
                                          &inner_workspace_size);
    if (status != CTC_STATUS_SUCCESS) {
        return status;
    }

    size_t offset = 0;
    char* ws = static_cast<char*>(workspace);

    void* inner_workspace = ws + offset;
    offset = align_bytes(offset + inner_workspace_size);

    int* major_offsets_dev = reinterpret_cast<int*>(ws + offset);
    offset = align_bytes(offset + static_cast<size_t>(num_majors + 1) * sizeof(int));

    int* major_flat_subclasses_dev = reinterpret_cast<int*>(ws + offset);

    cudaError_t cuda_status = cudaMemcpyAsync(
        major_offsets_dev, major_offsets,
        static_cast<size_t>(num_majors + 1) * sizeof(int),
        cudaMemcpyHostToDevice, options.stream);
    if (cuda_status != cudaSuccess) {
        return CTC_STATUS_MEMOPS_FAILED;
    }

    cuda_status = cudaMemcpyAsync(
        major_flat_subclasses_dev, major_flat_subclasses,
        static_cast<size_t>(major_offsets[num_majors]) * sizeof(int),
        cudaMemcpyHostToDevice, options.stream);
    if (cuda_status != cudaSuccess) {
        return CTC_STATUS_MEMOPS_FAILED;
    }

    if (gradients != nullptr) {
        cuda_status = cudaMemsetAsync(
            gradients, 0,
            static_cast<size_t>(maxT) * minibatch * alphabet_size * sizeof(float),
            options.stream);
        if (cuda_status != cudaSuccess) {
            return CTC_STATUS_MEMOPS_FAILED;
        }
    }

    GpuCTC<float> ctc(alphabet_size, minibatch, inner_workspace, options.stream,
                      options.blank_label, major_offsets_dev,
                      major_flat_subclasses_dev, num_majors,
                      major_label_offset);

    if (gradients != nullptr)
        return ctc.cost_and_grad(activations, gradients, costs,
                                 flat_labels, label_lengths, input_lengths);

    return ctc.score_forward(activations, costs,
                             flat_labels, label_lengths, input_lengths);
}

}
