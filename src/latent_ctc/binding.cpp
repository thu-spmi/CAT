/*
* Copyright 2016 SeanNaren (https://github.com/SeanNaren/warp-ctc)
*           2018-2019 Tsinghua University, Author: Hongyu Xiang, Hu Juntao (hujuntao_123@outlook.com)
*           2021-2022 Tsinghua University, Author: Huahuan Zheng
* Apache 2.0.
* Pytorch binding for CTC-CRF
*/

#include "gpu_ctc/ctc.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/Functions.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstdint>

void gpu_ctc(torch::Tensor probs,
             torch::Tensor grads,
             torch::Tensor labels,
             torch::Tensor label_sizes,
             torch::Tensor sizes,
             int minibatch_size,
             torch::Tensor costs,
             int blank_label)
{
    float *grads_ptr = grads.storage() ? grads.data_ptr<float>() : NULL;

    int probs_size = probs.size(2);

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.blank_label = blank_label;
    options.stream = c10::cuda::getCurrentCUDAStream(probs.device().index());

    size_t gpu_size_bytes;
    get_workspace_size(label_sizes.data_ptr<int>(), sizes.data_ptr<int>(),
                       probs_size, minibatch_size,
                       options, &gpu_size_bytes);

    auto gpu_workspace = torch::empty(
        {static_cast<int64_t>(gpu_size_bytes / sizeof(float))},
        torch::dtype(torch::kFloat32).device(probs.device()));

    compute_ctc_loss(probs.data_ptr<float>(), grads_ptr,
                     labels.data_ptr<int>(), label_sizes.data_ptr<int>(),
                     sizes.data_ptr<int>(), probs_size,
                     minibatch_size, costs.data_ptr<float>(),
                     (void *)gpu_workspace.data_ptr<float>(), options);

}

void gpu_latent_ctc(torch::Tensor probs,
                    torch::Tensor grads,
                    torch::Tensor labels,
                    torch::Tensor label_sizes,
                    torch::Tensor sizes,
                    int minibatch_size,
                    torch::Tensor costs,
                    int blank_label,
                    torch::Tensor major_offsets,
                    torch::Tensor major_flat_subclasses,
                    int major_label_offset)
{
    TORCH_CHECK(probs.is_cuda(), "probs must be a CUDA tensor");
    TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
    TORCH_CHECK(!labels.is_cuda(), "labels must be a CPU tensor");
    TORCH_CHECK(!label_sizes.is_cuda(), "label_sizes must be a CPU tensor");
    TORCH_CHECK(!sizes.is_cuda(), "sizes must be a CPU tensor");
    TORCH_CHECK(!costs.is_cuda(), "costs must be a CPU tensor");
    TORCH_CHECK(!major_offsets.is_cuda(), "major_offsets must be a CPU tensor");
    TORCH_CHECK(!major_flat_subclasses.is_cuda(), "major_flat_subclasses must be a CPU tensor");
    TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
    TORCH_CHECK(grads.scalar_type() == torch::kFloat32, "grads must be float32");
    TORCH_CHECK(labels.scalar_type() == torch::kInt32, "labels must be int32");
    TORCH_CHECK(label_sizes.scalar_type() == torch::kInt32, "label_sizes must be int32");
    TORCH_CHECK(sizes.scalar_type() == torch::kInt32, "sizes must be int32");
    TORCH_CHECK(costs.scalar_type() == torch::kFloat32, "costs must be float32");
    TORCH_CHECK(major_offsets.scalar_type() == torch::kInt32, "major_offsets must be int32");
    TORCH_CHECK(major_flat_subclasses.scalar_type() == torch::kInt32, "major_flat_subclasses must be int32");

    float *grads_ptr = grads.storage() ? grads.data_ptr<float>() : NULL;
    int probs_size = probs.size(2);
    int num_majors = major_offsets.size(0) - 1;

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.blank_label = blank_label;
    options.stream = c10::cuda::getCurrentCUDAStream(probs.device().index());

    size_t gpu_size_bytes;
    ctcStatus_t status = get_latent_workspace_size(
        label_sizes.data_ptr<int>(), sizes.data_ptr<int>(),
        probs_size, minibatch_size, options,
        major_offsets.data_ptr<int>(), num_majors, &gpu_size_bytes);
    TORCH_CHECK(status == CTC_STATUS_SUCCESS, "get_latent_workspace_size failed: ",
                ctcGetStatusString(status));

    auto gpu_workspace = torch::empty({static_cast<long>(gpu_size_bytes)},
                                      torch::dtype(torch::kUInt8).device(probs.device()));

    status = compute_latent_ctc_loss(
        probs.data_ptr<float>(), grads_ptr,
        labels.data_ptr<int>(), label_sizes.data_ptr<int>(),
        sizes.data_ptr<int>(), probs_size, minibatch_size,
        costs.data_ptr<float>(), (void *)gpu_workspace.data_ptr<uint8_t>(),
        options, major_offsets.data_ptr<int>(),
        major_flat_subclasses.data_ptr<int>(), num_majors,
        major_label_offset);
    TORCH_CHECK(status == CTC_STATUS_SUCCESS, "compute_latent_ctc_loss failed: ",
                ctcGetStatusString(status));
}

// pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gpu_ctc", &gpu_ctc, "CTC gpu_ctc");
    m.def("gpu_latent_ctc", &gpu_latent_ctc, "Latent CTC gpu_latent_ctc");
}
