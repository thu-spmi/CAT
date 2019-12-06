#include "THC.h"

void THNN_CudaBatchnormUpdateOutput(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *output_,
  THCTensor *weight_, THCTensor *bias_, THCTensor *runningMean_,
  THCTensor *runningVar_, THCTensor *saveMean_, THCTensor *saveStd_,
  int length_sum, int train, double momentum, double eps);

void THNN_CudaBatchnormMean(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *saveMean_, int length_sum);

void THNN_CudaBatchnormVar(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *saveMean_, THCTensor *saveVar_, int length_sum);

void THNN_CudaBatchnormBackward(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *gradOutput_,
  THCTensor *gradOutputMean_, THCTensor *dotP,
  THCTensor *gradInput_, THCTensor *gradWeight_, THCTensor *gradBias_,
  THCTensor *weight_, THCTensor *runningMean_, THCTensor *runningVar_,
  THCTensor *saveMean_, THCTensor *saveStd_, int length_sum, int train, double scale, double eps);

void THNN_CudaBatchnormGradStats(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *gradOutput_,
  THCTensor *runningMean_, THCTensor *saveMean_,
  THCTensor *gradOutputMean_, THCTensor *dotP_, int length_sum, int train);