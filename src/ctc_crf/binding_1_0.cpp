/*
*Copyright 2018-2019 Tsinghua University, Author: Hu Juntao (hujuntao_123@outlook.com)
*Apache 2.0.
*This code file provides binding functions so that the ctc function can be imported into python scripts.
*/

#include <TH.h>
#include <THC.h>
#include <THCTensor.h>
#include <iostream>
#include <algorithm>
#include "gpu_ctc/ctc.h"
#include "binding_1_0.h"

extern THCState *state;
extern int DEN_NUM_ARCS;
extern int DEN_NUM_STATES;

#undef ATOMIC_CONST
#define ATOMIC_CONST 32

extern "C" {
void Init(const char * fst_name, int n_gpus, int * gpus);

void Release(int n_gpus, int *gpus);

void compute_alpha(float *alpha,
                   float *logits,
                   const int batch_size,
                   int T,
                   const int alpha_size,
                   int logits_size,
                   int *input_lengths,
                   float * loglikelihood,
                   cudaStream_t stream);

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
                           cudaStream_t stream);
}

void init_env(const char * fst_name, torch::Tensor gpus) {
    int *gpus_ptr = (int*)gpus.data_ptr();
    int n_gpus = gpus.size(0);
    Init(fst_name, n_gpus, gpus_ptr);
}

void release_env(torch::Tensor gpus) {
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
    float *logits_ptr = (float*)logits.data_ptr();
    float *grad_net_ptr = (float*)grad_net.data_ptr();
    int *input_lengths_ptr = (int*)input_lengths.data_ptr();
    float *costs_alpha_ptr = (float*)costs_alpha.data_ptr();
    float *costs_beta_ptr = (float*)costs_beta.data_ptr();
    
    int logits_size = logits.size(2);
    int T = logits.size(1);
    int batch_size = logits.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    float *alpha = (float*)THCudaMalloc(state, sizeof(float)*(T+1)*batch_size*DEN_NUM_STATES);
    float *beta = (float*)THCudaMalloc(state, sizeof(float)*2*batch_size*DEN_NUM_STATES);
    float *grad_storage = (float*)THCudaMalloc(state, sizeof(float)*ATOMIC_CONST*batch_size*logits_size);

    // std::cout << logits_size << " " << T << " " << batch_size << std::endl;

    compute_alpha(alpha, logits_ptr, batch_size, T, DEN_NUM_STATES, logits_size, input_lengths_ptr, costs_alpha_ptr, stream);
    compute_beta_and_grad(beta, alpha, logits_ptr, costs_alpha_ptr, grad_storage, grad_net_ptr, batch_size, T,
        DEN_NUM_STATES, logits_size, input_lengths_ptr, costs_beta_ptr, stream);

    THCudaFree(state, (void*)alpha);
    THCudaFree(state, (void*)beta);
    THCudaFree(state, (void*)grad_storage);
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
    float *probs_ptr = (float *)probs.data_ptr(); 
    float* grads_ptr = grads.storage() ? (float*)grads.data_ptr() : NULL;   

    int *sizes_ptr = (int*)sizes.data_ptr();
    int *labels_ptr = (int*)labels.data_ptr();
    int *label_sizes_ptr = (int*)label_sizes.data_ptr();
    float *costs_ptr = (float*)costs.data_ptr();

    int probs_size = probs.size(2);

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.blank_label = blank_label;
    options.stream = at::cuda::getCurrentCUDAStream();

    size_t gpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes_ptr,
                       probs_size, minibatch_size,
                       options, &gpu_size_bytes);

    void* gpu_workspace = THCudaMalloc(state, gpu_size_bytes);

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes_ptr, probs_size,
                     minibatch_size, costs_ptr,
                     gpu_workspace, options);

    THCudaFree(state, (void *) gpu_workspace);
}

// pybind11 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gpu_den", &gpu_den, "CTC CRF gpu_den");
  m.def("init_env", &init_env, "CTC CRF init_env");
  m.def("release_env", &release_env, "CTC CRF release_env");
  m.def("gpu_ctc",&gpu_ctc,"CTC CRF gpu_ctc");
}
