#include "batchnorms_cuda_kernel.h"


extern THCState *state;

void BatchnormUpdateOutput(
  THTensor *input, THTensor *input_lengths, THTensor *output,
  THTensor *weight, THTensor *bias,
  THTensor *running_mean, THTensor *running_var,
  THTensor *save_mean, THTensor *save_std,
  int length_sum, int train, double momentum, double eps) {
  THNN_CudaBatchnormUpdateOutput(
    state, input, input_lengths, output, weight, bias, running_mean, running_var,
    save_mean, save_std, length_sum, train, momentum, eps);
}

void BatchnormMean(
  THTensor *input, THTensor *input_lengths, THTensor *save_mean, int length_sum) {
  THNN_CudaBatchnormMean(state, input, input_lengths, save_mean, length_sum);
}


void BatchnormVar(
  THTensor *input, THTensor *input_lengths, THTensor *save_mean, THTensor *save_var, int length_sum) {
  THNN_CudaBatchnormVar(
    state, input, input_lengths, save_mean, save_var, length_sum);
}


void BatchnormBackward(
  THTensor *input, THTensor *input_lengths, THTensor *gradOutput,
  THTensor *gradOutputMean, THTensor *dotP,
  THTensor *gradInput,
  THTensor *gradWeight, THTensor *gradBias, THTensor *weight,
  THTensor *running_mean, THTensor *running_var,
  THTensor *save_mean, THTensor *save_std,
  int length_sum, int train, double scale, double eps) {
  THNN_CudaBatchnormBackward(
      state, input, input_lengths, gradOutput, gradOutputMean, dotP,
      gradInput, gradWeight, gradBias, weight,
      running_mean, running_var, save_mean, save_std, length_sum, train, scale, eps);
}

void BatchnormGradStats(
  THTensor *input, THTensor *input_lengths, THTensor *gradOutput,
  THTensor *runningMean, THTensor *saveMean,
  THTensor *gradOutputMean, THTensor *dotP, int length_sum, int train) {
  THNN_CudaBatchnormGradStats(
    state, input, input_lengths, gradOutput, runningMean, saveMean,
    gradOutputMean, dotP, length_sum, train);
}
