#include <cstddef>
#include <iostream>
#include <algorithm>
#include "ctc.h"
#include "gpu_ctc.h"


extern "C" {

const char* ctcGetStatusString(ctcStatus_t status) {
    switch (status) {
    case CTC_STATUS_SUCCESS:
        return "no error";
    case CTC_STATUS_MEMOPS_FAILED:
        return "cuda memcpy or memset failed";
    case CTC_STATUS_INVALID_VALUE:
        return "invalid value";
    case CTC_STATUS_EXECUTION_FAILED:
        return "execution failed";

    case CTC_STATUS_UNKNOWN_ERROR:
    default:
        return "unknown error";

    }
}


ctcStatus_t compute_ctc_loss(const float* const activations,
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float *costs,
                             void *workspace,
                             ctcOptions options) {

    if (activations == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        workspace == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return CTC_STATUS_INVALID_VALUE;

        GpuCTC<float> ctc(alphabet_size, minibatch, workspace, options.stream,
                          options.blank_label);

        if (gradients != NULL)
            return ctc.cost_and_grad(activations, gradients, costs,
                                     flat_labels, label_lengths,
                                     input_lengths);
        else
            return ctc.score_forward(activations, costs, flat_labels,
                                     label_lengths, input_lengths);
}


ctcStatus_t get_workspace_size(const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size, int minibatch,
                               ctcOptions options,
                               size_t* size_bytes)
{
    if (label_lengths == nullptr ||
        input_lengths == nullptr ||
        size_bytes == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return CTC_STATUS_INVALID_VALUE;

    // This is the max of all S and T for all examples in the minibatch.
    int maxL = *std::max_element(label_lengths, label_lengths + minibatch);
    int maxT = *std::max_element(input_lengths, input_lengths + minibatch);

    const int S = 2 * maxL + 1;

    *size_bytes = 0;

        //nll_forward, nll_backward
    *size_bytes += 2 * sizeof(float) * minibatch;

    //repeats
    *size_bytes += sizeof(int) * minibatch;

    //label offsets
    *size_bytes += sizeof(int) * minibatch;

    //utt_length
    *size_bytes += sizeof(int) * minibatch;

    //label lengths
    *size_bytes += sizeof(int) * minibatch;

    //labels without blanks - overallocate for now
    *size_bytes += sizeof(int) * maxL * minibatch;

    //labels with blanks
    *size_bytes += sizeof(int) * S * minibatch;

    //alphas
    *size_bytes += sizeof(float) * S * maxT * minibatch;

    return CTC_STATUS_SUCCESS;
}
}
