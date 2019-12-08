void BatchnormUpdateOutput(
  THCudaTensor *input, THCudaIntTensor *input_lengths, THCudaTensor *output,
  THCudaTensor *weight, THCudaTensor *bias,
  THCudaTensor *running_mean, THCudaTensor *running_var,
  THCudaTensor *save_mean, THCudaTensor *save_std,
  int length_sum, int train, double momentum, double eps);


void BatchnormMean(
  THCudaTensor *input, THCudaIntTensor *input_lengths, THCudaTensor *save_mean, int length_sum);


void BatchnormVar(
  THCudaTensor *input, THCudaIntTensor *input_lengths, THCudaTensor *save_mean,
    THCudaTensor *save_var, int length_sum);

void BatchnormBackward(
  THCudaTensor *input, THCudaIntTensor *input_lengths, THCudaTensor *gradOutput,
  THCudaTensor *gradOutputMean, THCudaTensor *dotP,
  THCudaTensor *gradInput,
  THCudaTensor *gradWeight, THCudaTensor *gradBias, THCudaTensor *weight,
  THCudaTensor *running_mean, THCudaTensor *running_var,
  THCudaTensor *save_mean, THCudaTensor *save_std,
  int length_sum, int train, double scale, double eps);

void BatchnormGradStats(
  THCudaTensor *input, THCudaIntTensor *input_lengths, THCudaTensor *gradOutput,
  THCudaTensor *runningMean, THCudaTensor *saveMean,
  THCudaTensor *gradOutputMean, THCudaTensor *dotP, int length_sum, int train);