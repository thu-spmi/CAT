// Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang
// Apache 2.0.
// pytorch binding for CTC-CRF

void gpu_den(THCudaTensor *logits,
             THCudaTensor *grad_net,
             THCudaIntTensor *input_lengths,
             THCudaTensor *costs_alpha,
             THCudaTensor *costs_beta);

void init_env(const char * fst_name, THIntTensor *gpus);

void release_env(THIntTensor *gpus);

void gpu_ctc(THCudaTensor *probs,
             THCudaTensor *grads,
             THIntTensor *labels_ptr,
             THIntTensor *label_sizes_ptr,
             THIntTensor *sizes,
             int minibatch_size,
             THFloatTensor *costs,
             int blank_label);