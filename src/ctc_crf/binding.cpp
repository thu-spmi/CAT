/*
* Copyright 2016 SeanNaren (https://github.com/SeanNaren/warp-ctc)
*           2018-2019 Tsinghua University, Author: Hongyu Xiang, Hu Juntao (hujuntao_123@outlook.com)
*           2021-2022 Tsinghua University, Author: Huahuan Zheng
* Apache 2.0.
* Pytorch binding for CTC-CRF
*/

#include "gpu_ctc/ctc.h"
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <torch/extension.h>

extern int DEN_NUM_ARCS;
extern int DEN_NUM_STATES;

#undef ATOMIC_CONST
#define ATOMIC_CONST 32

extern "C"
{
    void Init(const char *fst_name, int n_gpus, int *gpus);

    void Release(int n_gpus, int *gpus);

    void compute_alpha(float *alpha,
                       float *logits,
                       const int batch_size,
                       int T,
                       const int alpha_size,
                       int logits_size,
                       int *input_lengths,
                       float *loglikelihood,
                       cudaStream_t stream);

    void compute_beta_and_grad(float *beta,
                               const float *const alpha,
                               const float *const logits,
                               const float *const alpha_lld,
                               float *grad_storage,
                               float *grad_net,
                               const int batch_size,
                               const int T,
                               const int beta_size,
                               const int logits_size,
                               const int *const input_lengths,
                               float *loglikelihood,
                               cudaStream_t stream);
}

void init_env(const char *fst_name, torch::Tensor gpus)
{
    int *gpus_ptr = (int *)gpus.data_ptr();
    int n_gpus = gpus.size(0);
    Init(fst_name, n_gpus, gpus_ptr);
}

void release_env(torch::Tensor gpus)
{
    int *gpus_ptr = (int *)gpus.data_ptr();
    int n_gpus = gpus.size(0);
    Release(n_gpus, gpus_ptr);
}

void gpu_den(torch::Tensor logits,
             torch::Tensor grad_net,
             torch::Tensor input_lengths,
             torch::Tensor costs_alpha,
             torch::Tensor costs_beta)
{
    int logits_size = logits.size(2);
    int T = logits.size(1);
    int batch_size = logits.size(0);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.device().index());

    auto alpha = torch::empty({T+1, batch_size, DEN_NUM_STATES}, torch::dtype(torch::kFloat32).device(logits.device()));
    auto beta = torch::empty({2, batch_size, DEN_NUM_STATES}, torch::dtype(torch::kFloat32).device(logits.device()));
    auto grad_storage = torch::empty({ATOMIC_CONST, batch_size, logits_size}, torch::dtype(torch::kFloat32).device(logits.device()));

    compute_alpha(alpha.data_ptr<float>(), logits.data_ptr<float>(), batch_size, T, DEN_NUM_STATES, logits_size, input_lengths.data_ptr<int>(), costs_alpha.data_ptr<float>(), stream);
    compute_beta_and_grad(beta.data_ptr<float>(), alpha.data_ptr<float>(), logits.data_ptr<float>(), costs_alpha.data_ptr<float>(), grad_storage.data_ptr<float>(), grad_net.data_ptr<float>(), batch_size, T,
                          DEN_NUM_STATES, logits_size, input_lengths.data_ptr<int>(), costs_beta.data_ptr<float>(), stream);
}

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

    auto gpu_workspace = torch::empty({gpu_size_bytes/4}, torch::dtype(torch::kFloat32).device(probs.device()));

    compute_ctc_loss(probs.data_ptr<float>(), grads_ptr,
                     labels.data_ptr<int>(), label_sizes.data_ptr<int>(),
                     sizes.data_ptr<int>(), probs_size,
                     minibatch_size, costs.data_ptr<float>(),
                     (void *)gpu_workspace.data_ptr<float>(), options);

}

// pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gpu_den", &gpu_den, "CTC CRF gpu_den");
    m.def("init_env", &init_env, "CTC CRF init_env");
    m.def("release_env", &release_env, "CTC CRF release_env");
    m.def("gpu_ctc", &gpu_ctc, "CTC CRF gpu_ctc");
}