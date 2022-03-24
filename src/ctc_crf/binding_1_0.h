/*
*Copyright 2018-2019 Tsinghua University, Author: Hu Juntao (hujuntao_123@outlook.com)
*Apache 2.0.
*This code file provides binding functions so that the ctc function can be imported into python scripts.
*/

#include <torch/extension.h>

void gpu_den(torch::Tensor logits,
             torch::Tensor grad_net,
             torch::Tensor input_lengths,
             torch::Tensor costs_alpha,
             torch::Tensor costs_beta);

void init_env(const char * fst_name, torch::Tensor gpus);

void release_env(torch::Tensor gpus);

void gpu_ctc(torch::Tensor probs,
             torch::Tensor grads,
             torch::Tensor labels_ptr,
             torch::Tensor label_sizes_ptr,
             torch::Tensor sizes,
             int minibatch_size,
             torch::Tensor costs,
             int blank_label);
