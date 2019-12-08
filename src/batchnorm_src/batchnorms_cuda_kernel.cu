#include "TH.h"
#include "THCUNN.h"
#include "common.h"
#include "THCNumerics.cuh"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
const int WARP_SIZE = 32;

// The maximum number of threads in a block
const int MAX_BLOCK_SIZE = 512;

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template <typename Dtype, typename Acctype>
struct Float2 {
  Acctype v1, v2;
  __device__ Float2() {}
  __device__ Float2(Dtype v1, Dtype v2) : v1(ScalarConvert<Dtype, Acctype>::to(v1)), v2(ScalarConvert<Dtype, Acctype>::to(v2)) {}
  __device__ Float2(Dtype v) : v1(ScalarConvert<Dtype, Acctype>::to(v)), v2(ScalarConvert<Dtype, Acctype>::to(v)) {}
  __device__ Float2(int v) : v1(ScalarConvert<int, Acctype>::to(v)), v2(ScalarConvert<int, Acctype>::to(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template <typename Dtype, typename Acctype, typename DeviceTensor3>
struct SumOp {
  __device__ SumOp(const DeviceTensor3 t) : tensor(t) {}
  __device__ __forceinline__ Acctype operator()(int batch, int plane, int n) {
    return ScalarConvert<Dtype, Acctype>::to(tensor[batch][plane][n]);
  }
  const DeviceTensor3 tensor;
};

template <typename Dtype, typename Acctype, typename DeviceTensor3>
struct VarOp {
  __device__ VarOp(Acctype m, const DeviceTensor3 t) : mean(m), tensor(t) {}
  __device__ __forceinline__ Acctype operator()(int batch, int plane, int n) {
    Dtype val = tensor[batch][plane][n];
    return (val - mean) * (val - mean);
  }
  const Acctype mean;
  const DeviceTensor3 tensor;
};

template <typename Dtype, typename Acctype, typename DeviceTensor3>
struct GradOp {
  __device__ GradOp(Acctype m, const DeviceTensor3 i, const DeviceTensor3 g)
    : mean(m), input(i), gradOutput(g) {}
  __device__ __forceinline__ Float2<Dtype, Acctype> operator()(int batch, int plane, int n) {
    Dtype g = gradOutput[batch][plane][n];
    Dtype c = ScalarConvert<Acctype, Dtype>::to(input[batch][plane][n] - mean);
    return Float2<Dtype, Acctype>(g, g * c);
  }
  const Acctype mean;
  const DeviceTensor3 input;
  const DeviceTensor3 gradOutput;
};

// Sum across all threads within a warp
template <typename T>
static __device__ __forceinline__ T warpSum(T val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
  }
#else
  __shared__ T values[MAX_BLOCK_SIZE];
  values[threadIdx.x] = val;
  __threadfence_block();
  const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  for (int i = 1; i < WARP_SIZE; i++) {
    val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
  }
#endif
  return val;
}

template <typename Dtype, typename Acctype>
static __device__ __forceinline__ Float2<Dtype, Acctype> warpSum(Float2<Dtype, Acctype> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

template<typename T, typename Op, typename DeviceTensor3, typename IndexTensor>
__device__ T reduce_vl(Op op, DeviceTensor3 tensor, int plane, IndexTensor input_lengths) {
  T sum = (T)0;
  for (int batch = 0; batch < tensor.getSize(0); ++batch) {
    for (int x = threadIdx.x; x < input_lengths[batch]; x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];
  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    shared[threadIdx.x / WARP_SIZE] = sum;
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T)0;
  }
  __syncthreads();
  if (threadIdx.x / WARP_SIZE == 0) {
    sum = warpSum(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

template <typename Dtype, typename Acctype, typename DeviceTensor1, typename DeviceTensor3, typename IndexTensor>
__global__ void batchnorm_inference_kernel(
    const DeviceTensor3 input,
    const IndexTensor input_lengths,
    DeviceTensor3 output,
    DeviceTensor1 runningMean,
    DeviceTensor1 runningVar,
    const DeviceTensor1 weight,
    const DeviceTensor1 bias,
    Acctype epsilon) {

  int plane = blockIdx.x;

  Acctype invstd = Acctype(1) / sqrt(runningVar[plane].ldg() + epsilon);
  Acctype mean = ScalarConvert<Dtype, Acctype>::to(runningMean[plane].ldg());
  Acctype gamma = weight.numElements() > 0 ? ScalarConvert<Dtype, Acctype>::to(weight[plane].ldg()) : Acctype(1);
  Acctype beta = bias.numElements() > 0 ? ScalarConvert<Dtype, Acctype>::to(bias[plane].ldg()) : Acctype(0);

  // Write normalized and update the output
  for (int batch = 0; batch < input.getSize(0); batch++) {
    for (int x = threadIdx.x; x < input_lengths[batch]; x += blockDim.x) {
      Dtype inp = input[batch][plane][x].ldg();
      output[batch][plane][x] = ScalarConvert<Acctype, Dtype>::to(gamma * (inp - mean) * invstd + beta);
    }
  }
}

template <typename Dtype, typename Acctype, typename DeviceTensor1, typename DeviceTensor3, typename IndexTensor>
__global__ void batchnorm_mean_kernel(
    const DeviceTensor3 input,
    const IndexTensor input_lengths,
    const int length_sum,
    DeviceTensor1 out_mean) {
  int plane = blockIdx.x;
  Acctype norm = Acctype(1) / length_sum;
  Acctype mean = reduce_vl<Acctype>(SumOp<Dtype, Acctype, DeviceTensor3>(input), input, plane, input_lengths) * norm;

  if (threadIdx.x == 0) {
    out_mean[plane] = ScalarConvert<Acctype, Dtype>::to(mean);
  }
}


template <typename Dtype, typename Acctype, typename DeviceTensor1, typename DeviceTensor3, typename IndexTensor>
__global__ void batchnorm_val_kernel(
    const DeviceTensor3 input,
    const IndexTensor input_lengths,
    const int length_sum,
    const DeviceTensor1 in_mean,
    DeviceTensor1 out_var) {
  int plane = blockIdx.x;
  Acctype norm = Acctype(1) / length_sum;
  Acctype mean = ScalarConvert<Dtype, Acctype>::to(in_mean[plane]);
  Acctype var = reduce_vl<Acctype>(VarOp<Dtype, Acctype, DeviceTensor3>(mean, input), input, plane, input_lengths) * norm;
  if (threadIdx.x == 0) {
    out_var[plane] = ScalarConvert<Acctype, Dtype>::to(var);
  }
}

template <typename Dtype, typename Acctype, typename DeviceTensor1, typename DeviceTensor3, typename IndexTensor>
__global__ void batchnorm_output_kernel(
    const DeviceTensor3 input,
    const IndexTensor input_lengths,
    const int length_sum,
    DeviceTensor3 output,
    const DeviceTensor1 weight,
    const DeviceTensor1 bias,
    const Acctype epsilon,
    const Acctype momentum,
    DeviceTensor1 runningMean,
    DeviceTensor1 runningVar,
    DeviceTensor1 saveMean,
    DeviceTensor1 saveVar) {

  int plane = blockIdx.x;
  int N = length_sum;

  Acctype mean = ScalarConvert<Dtype, Acctype>::to(saveMean[plane]);
  Acctype var = ScalarConvert<Dtype, Acctype>::to(saveVar[plane]);
  Acctype invStd = 1 / sqrt(var + epsilon);

  // Save the mean, variance, and moving averages
  if (threadIdx.x == 0) {
    // Momentum based writeback
    Acctype unbiasedVar = var * N / (N - 1);
    runningMean[plane] = ScalarConvert<Acctype, Dtype>::to((1 - momentum) * runningMean[plane] + momentum * mean);
    runningVar[plane] = ScalarConvert<Acctype, Dtype>::to((1 - momentum) * runningVar[plane] + momentum * unbiasedVar);
  }

  // Write normalized and update the output
  Acctype gamma = weight.numElements() > 0 ? ScalarConvert<Dtype, Acctype>::to(weight[plane]) : ScalarConvert<int, Acctype>::to(1);
  Acctype beta = bias.numElements() > 0 ? ScalarConvert<Dtype, Acctype>::to(bias[plane]) : ScalarConvert<int, Acctype>::to(0);
  for (int batch = 0; batch < input.getSize(0); ++batch) {
    for (int x = threadIdx.x; x < input_lengths[batch]; x += blockDim.x) {
    //for (int x = threadIdx.x; x < input.getSize(2); x += blockDim.x) {
      // int t = input_lengths[batch];
      // printf("block: %d, batch: %d, input_length: %d, x:%d\n", blockIdx.x, batch, t, x);
      Dtype inp = input[batch][plane][x].ldg();
      output[batch][plane][x] = ScalarConvert<Acctype, Dtype>::to(gamma * (inp - mean) * invStd + beta);
    }
  }
}


template <typename Dtype, typename Acctype, typename DeviceTensor1, typename DeviceTensor3, typename IndexTensor>
__global__ void batchnorm_grad_stats_kernel(
    const DeviceTensor3 input,
    const IndexTensor input_lengths,
    const int length_sum,
    const DeviceTensor3 gradOutput,
    const DeviceTensor1 runningMean,
    const DeviceTensor1 saveMean,
    DeviceTensor1 gradOutputMean_all,
    DeviceTensor1 dotP_all,
    bool train) {
  int plane = blockIdx.x;
  int N = length_sum;

  Acctype mean;
  if (train) {
    mean = ScalarConvert<Dtype, Acctype>::to(saveMean[plane]);
  } else {
    mean = ScalarConvert<Dtype, Acctype>::to(runningMean[plane]);
  }

  Acctype norm = Acctype(1) / N;
  GradOp<Dtype, Acctype, DeviceTensor3> g(mean, input, gradOutput);
  Float2<Dtype, Acctype> res = reduce_vl<Float2<Dtype, Acctype>, GradOp<Dtype, Acctype, DeviceTensor3>, DeviceTensor3>
    (g, gradOutput, plane, input_lengths);

  Acctype gradOutputMean = res.v1 * norm;
  Acctype dotP = res.v2 * norm;

  if (threadIdx.x == 0) {
    gradOutputMean_all[plane] = ScalarConvert<Acctype, Dtype>::to(gradOutputMean);
    dotP_all[plane] = ScalarConvert<Acctype, Dtype>::to(dotP);
  }
}

template <typename Dtype, typename Acctype, typename DeviceTensor1, typename DeviceTensor3, typename IndexTensor>
__global__ void batchnorm_backward_kernel(
    const DeviceTensor3 input,
    const IndexTensor input_lengths,
    const int length_sum,
    const DeviceTensor3 gradOutput,
    const DeviceTensor1 gradOutputMean,
    const DeviceTensor1 dotP_all,
    DeviceTensor3 gradInput,
    DeviceTensor1 gradWeight,
    DeviceTensor1 gradBias,
    const DeviceTensor1 weight,
    const DeviceTensor1 runningMean,
    const DeviceTensor1 runningVar,
    const DeviceTensor1 saveMean,
    const DeviceTensor1 saveVar,
    bool train,
    Acctype scale,
    double eps) {

  int plane = blockIdx.x;
  int N = length_sum;

  Acctype mean, stdVal;
  if (train) {
    mean = ScalarConvert<Dtype, Acctype>::to(saveMean[plane]);
    stdVal = 1 / sqrt(ScalarConvert<Dtype, Acctype>::to(saveVar[plane]) + eps);
  } else {
    mean = ScalarConvert<Dtype, Acctype>::to(runningMean[plane]);
    stdVal = 1 / sqrt(runningVar[plane] + eps);
  }

  Acctype weightVal = weight.numElements() > 0 ? ScalarConvert<Dtype, Acctype>::to(weight[plane]) : Acctype(1);

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(gradOutput)
  // 2. DotProduct(input - mean, gradOutput)
  // Acctype gradOutputSum = res.v1;
  Acctype gradOutputSum = ScalarConvert<Dtype, Acctype>::to(gradOutputMean[plane]) * N;
  Acctype dotP = ScalarConvert<Dtype, Acctype>::to(dotP_all[plane]);

  // Acctype gradMean = gradOutputSum * norm;
  Acctype gradMean = ScalarConvert<Dtype, Acctype>::to(gradOutputMean[plane]);
  // Acctype projScale = dotP * norm * stdVal * stdVal;
  Acctype projScale = dotP * stdVal * stdVal;
  Acctype gradScale = stdVal * weightVal;

  if (gradInput.numElements() > 0) {
    for (int batch = 0; batch < gradOutput.getSize(0); ++batch) {
      for (int x = threadIdx.x; x < input_lengths[batch]; x += blockDim.x) {
        Dtype gradOut = gradOutput[batch][plane][x];
        if (train) {
          Dtype inp = input[batch][plane][x];
          Acctype proj = (inp - mean) * projScale;
          gradInput[batch][plane][x] = ScalarConvert<Acctype, Dtype>::to((gradOut - proj - gradMean) * gradScale);
        } else {
          gradInput[batch][plane][x] = ScalarConvert<Acctype, Dtype>::to(gradOut * gradScale);
        }
      }
    }
  }

  if (gradWeight.numElements() > 0) {
    if (threadIdx.x == 0) {
      gradWeight[plane] += ScalarConvert<Acctype, Dtype>::to(scale * dotP * stdVal);
    }
  }

  if (gradBias.numElements() > 0) {
    if (threadIdx.x == 0) {
      gradBias[plane] += ScalarConvert<Acctype, Dtype>::to(scale * gradOutputSum);
    }
  }
}


#define FloatTensor3 THCDeviceTensor<float, 3>
#define FloatTensor1 THCDeviceTensor<float, 1>
#define IntTensor1 THCDeviceTensor<int, 1>

template <typename Dtype, int Dim>
static THCDeviceTensor<Dtype, Dim> devicetensor(THCState *state, THCTensor *t) {
  if (!t) {
    return THCDeviceTensor<Dtype, Dim>();
  }
  int inDim = t->dim();
  THAssert(inDim == Dim);
  return toDeviceTensor<Dtype, Dim>(state, t);
}

extern "C" void THNN_CudaBatchnormUpdateOutput(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *output_,
  THCTensor *weight_, THCTensor *bias_, THCTensor *runningMean_,
  THCTensor *runningVar_, THCTensor *saveMean_, THCTensor *saveStd_,
  int length_sum, int train, double momentum, double eps);

extern "C" void THNN_CudaBatchnormMean(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *saveMean_, int length_sum);

extern "C" void THNN_CudaBatchnormVar(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *saveMean_, THCTensor *saveVar_, int length_sum);


void THNN_CudaBatchnormMean(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *saveMean_, int length_sum) {
  FloatTensor3 input = devicetensor<float, 3>(state, input_);
  FloatTensor1 saveMean = devicetensor<float, 1>(state, saveMean_);
  IntTensor1 input_lengths = devicetensor<int, 1>(state, input_lengths_);

  cudaStream_t s = THCState_getCurrentStream(state);
  cudaDeviceProp *prop = THCState_getCurrentDeviceProperties(state);

  dim3 blocks(input.getSize(1));
  dim3 threads(getNumThreads(input.getSize(2)));
  batchnorm_mean_kernel<float, float, FloatTensor1, FloatTensor3, IntTensor1> <<<blocks, threads, 0, s>>>(
  input, input_lengths, length_sum, saveMean);
  THCudaCheck(cudaGetLastError());
}

void THNN_CudaBatchnormVar(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_,
    THCTensor *saveMean_, THCTensor *saveVar_, int length_sum) {

  FloatTensor3 input = devicetensor<float, 3>(state, input_);
  FloatTensor1 saveMean = devicetensor<float, 1>(state, saveMean_);
  FloatTensor1 saveVar = devicetensor<float, 1>(state, saveVar_);
  IntTensor1 input_lengths = devicetensor<int, 1>(state, input_lengths_);

  cudaStream_t s = THCState_getCurrentStream(state);
  cudaDeviceProp *prop = THCState_getCurrentDeviceProperties(state);

  dim3 blocks(input.getSize(1));
  dim3 threads(getNumThreads(input.getSize(2)));
  batchnorm_val_kernel<float, float, FloatTensor1, FloatTensor3, IntTensor1> <<<blocks, threads, 0, s>>>(
  input, input_lengths, length_sum, saveMean, saveVar);
  THCudaCheck(cudaGetLastError());
}

void THNN_CudaBatchnormUpdateOutput(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *output_,
  THCTensor *weight_, THCTensor *bias_, THCTensor *runningMean_,
  THCTensor *runningVar_, THCTensor *saveMean_, THCTensor *saveStd_,
  int length_sum, int train, double momentum, double eps) {

  THCTensor_resizeAs(state, output_, input_);
  FloatTensor3 input = devicetensor<float, 3>(state, input_);
  FloatTensor3 output = devicetensor<float, 3>(state, output_);
  FloatTensor1 weight = devicetensor<float, 1>(state, weight_);
  FloatTensor1 bias = devicetensor<float, 1>(state, bias_);
  FloatTensor1 runningMean = devicetensor<float, 1>(state, runningMean_);
  FloatTensor1 runningVar = devicetensor<float, 1>(state, runningVar_);
  FloatTensor1 saveMean = devicetensor<float, 1>(state, saveMean_);
  FloatTensor1 saveStd = devicetensor<float, 1>(state, saveStd_);
  IntTensor1 input_lengths = devicetensor<int, 1>(state, input_lengths_);

  cudaStream_t s = THCState_getCurrentStream(state);
  cudaDeviceProp *prop = THCState_getCurrentDeviceProperties(state);

  if (!train) {
    dim3 blocks(input.getSize(1));
    dim3 threads(getNumThreads(input.getSize(2)));
    batchnorm_inference_kernel<float, float, FloatTensor1, FloatTensor3, IntTensor1> <<<blocks, threads, 0, s>>>(
      input, input_lengths, output, runningMean, runningVar, weight, bias, eps);
  } else {
    dim3 blocks(input.getSize(1));
    dim3 threads(getNumThreads(input.getSize(2)));
    batchnorm_output_kernel<float, float, FloatTensor1, FloatTensor3, IntTensor1> <<<blocks, threads, 0, s>>>(
      input, input_lengths, length_sum, output, weight, bias, eps, momentum, runningMean, runningVar,
      saveMean, saveStd);
  }
  THCudaCheck(cudaGetLastError());
}

extern "C" void THNN_CudaBatchnormBackward(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *gradOutput_,
  THCTensor *gradOutputMean_, THCTensor *dotP,
  THCTensor *gradInput_, THCTensor *gradWeight_, THCTensor *gradBias_,
  THCTensor *weight_, THCTensor *runningMean_, THCTensor *runningVar_,
  THCTensor *saveMean_, THCTensor *saveStd_, int length_sum, int train, double scale, double eps);


extern "C" void THNN_CudaBatchnormGradStats(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *gradOutput_,
  THCTensor *runningMean_, THCTensor *saveMean_,
  THCTensor *gradOutputMean_, THCTensor *dotP_, int length_sum, int train);


void THNN_CudaBatchnormGradStats(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *gradOutput_,
  THCTensor *runningMean_, THCTensor *saveMean_,
  THCTensor *gradOutputMean_, THCTensor *dotP_, int length_sum, int train) {

  // THCUNN_check_shape(state, input_, gradOutput_);
  FloatTensor3 input = devicetensor<float, 3>(state, input_);
  FloatTensor3 gradOutput = devicetensor<float, 3>(state, gradOutput_);
  FloatTensor1 gradOutputMean = devicetensor<float, 1>(state, gradOutputMean_);
  FloatTensor1 dotP = devicetensor<float, 1>(state, dotP_);
  FloatTensor1 runningMean = devicetensor<float, 1>(state, runningMean_);
  FloatTensor1 saveMean = devicetensor<float, 1>(state, saveMean_);
  IntTensor1 input_lengths = devicetensor<int, 1>(state, input_lengths_);

  cudaStream_t s = THCState_getCurrentStream(state);

  dim3 blocks(gradOutput.getSize(1));
  dim3 threads(getNumThreads(gradOutput.getSize(2)));
  batchnorm_grad_stats_kernel<float,  float,  FloatTensor1, FloatTensor3, IntTensor1> <<<blocks, threads, 0, s>>>(
    input, input_lengths, length_sum, gradOutput, runningMean, saveMean, gradOutputMean, dotP, train);
  THCudaCheck(cudaGetLastError());
}


void THNN_CudaBatchnormBackward(
  THCState *state, THCTensor *input_, THCTensor *input_lengths_, THCTensor *gradOutput_,
  THCTensor *gradOutputMean_, THCTensor *dotP_,
  THCTensor *gradInput_, THCTensor *gradWeight_, THCTensor *gradBias_,
  THCTensor *weight_, THCTensor *runningMean_, THCTensor *runningVar_,
  THCTensor *saveMean_, THCTensor *saveStd_, int length_sum, int train, double scale, double eps) {

  // THCUNN_check_shape(state, input_, gradOutput_);
  FloatTensor3 input = devicetensor<float, 3>(state, input_);
  FloatTensor3 gradOutput = devicetensor<float, 3>(state, gradOutput_);
  FloatTensor1 gradOutputMean = devicetensor<float, 1>(state, gradOutputMean_);
  FloatTensor1 dotP = devicetensor<float, 1>(state, dotP_);
  FloatTensor3 gradInput = devicetensor<float, 3>(state, gradInput_);
  FloatTensor1 gradWeight = devicetensor<float, 1>(state, gradWeight_);
  FloatTensor1 gradBias = devicetensor<float, 1>(state, gradBias_);
  FloatTensor1 weight = devicetensor<float, 1>(state, weight_);
  FloatTensor1 runningMean = devicetensor<float, 1>(state, runningMean_);
  FloatTensor1 runningVar = devicetensor<float, 1>(state, runningVar_);
  FloatTensor1 saveMean = devicetensor<float, 1>(state, saveMean_);
  FloatTensor1 saveStd = devicetensor<float, 1>(state, saveStd_);
  IntTensor1 input_lengths = devicetensor<int, 1>(state, input_lengths_);

  cudaStream_t s = THCState_getCurrentStream(state);

  dim3 blocks(gradOutput.getSize(1));
  dim3 threads(getNumThreads(gradOutput.getSize(2)));
  batchnorm_backward_kernel<float,  float,  FloatTensor1, FloatTensor3, IntTensor1> <<<blocks, threads, 0, s>>>(
    input, input_lengths, length_sum, gradOutput, gradOutputMean, dotP, gradInput, gradWeight, gradBias, weight, runningMean, runningVar,
    saveMean, saveStd, train, scale, eps);
  THCudaCheck(cudaGetLastError());
}