// Copyright 2016 SeanNaren (https://github.com/SeanNaren/warp-ctc)
//           2018-2019 Tsinghua University, Author: Hongyu Xiang
// Apache 2.0.
// pytorch binding for CTC-CRF

#include <TH.h>
#include <THC.h>
#include <THCTensor.h>
#include <iostream>
#include <algorithm>
#include "gpu_ctc/ctc.h"

extern THCState *state;
extern int DEN_NUM_ARCS;
extern int DEN_NUM_STATES;

#undef ATOMIC_CONST
#define ATOMIC_CONST 32

extern "C" {
// void init_env(void);
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


void init_env(const char * fst_name, THIntTensor *gpus) {
    int *gpus_ptr = THIntTensor_data(gpus);
    int n_gpus = THIntTensor_size(gpus, 0);
    Init(fst_name, n_gpus, gpus_ptr);
}

void release_env(THIntTensor *gpus) {
    int *gpus_ptr = THIntTensor_data(gpus);
    int n_gpus = THIntTensor_size(gpus, 0);
    Release(n_gpus, gpus_ptr);
}

void gpu_den(THCudaTensor *logits,
             THCudaTensor *grad_net,
             THCudaIntTensor *input_lengths,
             THCudaTensor *costs_alpha,
             THCudaTensor *costs_beta)
{
    float *logits_ptr = THCudaTensor_data(state, logits);
    float *grad_net_ptr = THCudaTensor_data(state, grad_net);
    int *input_lengths_ptr = THCudaIntTensor_data(state, input_lengths);
    float *costs_alpha_ptr = THCudaTensor_data(state, costs_alpha);
    float *costs_beta_ptr = THCudaTensor_data(state, costs_beta);
    
    int logits_size = THCudaTensor_size(state, logits, 2);
    int T = THCudaTensor_size(state, logits, 1);
    int batch_size = THCudaTensor_size(state, logits, 0);

    cudaStream_t stream = THCState_getCurrentStream(state);

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



void gpu_ctc(THCudaTensor *probs,
             THCudaTensor *grads,
             THIntTensor *labels,
             THIntTensor *label_sizes,
             THIntTensor *sizes,
             int minibatch_size,
             THFloatTensor *costs,
             int blank_label)
{
    float *probs_ptr = THCudaTensor_data(state, probs);
    float *grads_ptr;
    if (THCudaTensor_storage(state, grads)) {
        grads_ptr = THCudaTensor_data(state, grads);
    } else {
        grads_ptr = NULL; // this will trigger the score forward code path
    }

    int *sizes_ptr = THIntTensor_data(sizes);
    int *labels_ptr = THIntTensor_data(labels);
    int *label_sizes_ptr = THIntTensor_data(label_sizes);
    float *costs_ptr = THFloatTensor_data(costs);

    int probs_size = THFloatTensor_size(probs, 2);

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.blank_label = blank_label;
    options.stream = THCState_getCurrentStream(state);

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
}
