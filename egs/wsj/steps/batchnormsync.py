import torch
import numpy as np
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.autograd import Function

from Queue import Queue
from threading import Condition
sys.path.append('../../src/batchnorm_src')
from batchnorm_utils import pytorch as batch_norm

cum_queue = Queue()
broadcast_queue = Queue()
broadcast_cv = Condition()

class BatchnormFunction(Function):
    def __init__(self, running_mean, running_var, training,
                 cum_queue, broadcast_queue, device_ids, sync,
                 eps=1e-5, momentum=0.1, affine=True):
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.running_mean = running_mean
        self.running_var = running_var
        self.mean = None
        self.var = None
        self.training = training
        self.cum_queue = cum_queue
        self.broadcast_queue = broadcast_queue
        self.device_ids = device_ids
        self.sync = sync

    #TODO when batchsize is small than GPU, the thread will hang
    def forward(self, input, input_lengths, weight, bias):
        output = input.new_zeros(input.size())
        self.save_for_backward(input, input_lengths, weight, bias)

        batch_size = int(input.size(0))
        device_ids = self.device_ids
        # print('device', input.get_device(), flush=True)

        # we only provide cuda version
        length_sum = torch.sum(input_lengths).item()
        # print("ids: ", self.device_ids)
        # print("device: ", input.get_device())
        # print("length_sum: ", length_sum)

        mean_cuda = input.new_zeros(input.size(1))
        var_cuda = input.new_ones(input.size(1))
        batch_norm.BatchnormMean(input, input_lengths, mean_cuda, length_sum)

        if len(device_ids) > 1 and self.sync and self.training:
            mean_cuda.copy_(torch.from_numpy(self.cum_mean(
                input.get_device(), mean_cuda.cpu().numpy(), length_sum)))

        batch_norm.BatchnormVar(input, input_lengths, mean_cuda, var_cuda, length_sum)

        if len(device_ids) > 1 and self.sync and self.training:
            var_cuda.copy_(torch.from_numpy(self.cum_mean(
                input.get_device(), var_cuda.cpu().numpy(), length_sum)))

        self.mean = mean_cuda
        self.var = var_cuda

        batch_norm.BatchnormUpdateOutput(
                input, input_lengths, output, weight, bias,
                self.running_mean, self.running_var, self.mean, self.var,
                length_sum, self.training, self.momentum, self.eps)
        return output


    ############################################## wrong ################################################
    def cum_mean(self, this_device, this_mean, item_size):
        cum_queue.put((item_size, this_mean))
        total_mean = np.zeros(this_mean.shape, dtype=np.float64)
        total_item_size = 0
        if this_device == self.device_ids[0]:
            for _ in self.device_ids:
                item = cum_queue.get()
                total_item_size += item[0]
                total_mean += item[0] * item[1]
                cum_queue.task_done()
            total_mean /= total_item_size
            broadcast_cv.acquire()
            for _ in range(len(self.device_ids) - 1):
                broadcast_queue.put(total_mean)
            broadcast_cv.notify_all()
            broadcast_cv.release()
        else:
            broadcast_cv.acquire()
            if broadcast_queue.qsize() == 0:
                broadcast_cv.wait()
            total_mean = broadcast_queue.get()
            broadcast_queue.task_done()
            broadcast_cv.release()
        # assert cum_queue.empty()
        broadcast_queue.join()
        return total_mean

    def backward(self, grad_output):
        input, input_lengths, weight, bias = self.saved_tensors
        grad_input = grad_output.new_zeros(input.size())
        grad_weight = grad_output.new_zeros(weight.size())
        grad_bias = grad_output.new_zeros(bias.size())

        batch_size = int(grad_output.size(0))
        grad_output_mean_cuda = grad_output.new_zeros(grad_output.size(1))
        dotP_cuda = grad_output.new_zeros(grad_output.size(1))

        length_sum = torch.sum(input_lengths).item()

        batch_norm.BatchnormGradStats(input, input_lengths, grad_output, self.running_mean,
                self.mean, grad_output_mean_cuda, dotP_cuda, length_sum, self.training)

        if len(self.device_ids) > 1 and self.sync:
            grad_output_mean_cuda.copy_(torch.from_numpy(
                self.cum_mean(grad_output.get_device(),
                              grad_output_mean_cuda.cpu().numpy(),
                              length_sum)))
            dotP_cuda.copy_(torch.from_numpy(
                self.cum_mean(grad_output.get_device(),
                              dotP_cuda.cpu().numpy(),
                              length_sum)))

        batch_norm.BatchnormBackward(
            input, input_lengths, grad_output, grad_output_mean_cuda, dotP_cuda,
            grad_input, grad_weight, grad_bias,
            weight, self.running_mean, self.running_var,
            self.mean, self.var, length_sum, self.training, 1.0, self.eps)
        return grad_input, None, grad_weight, grad_bias


class BatchnormSync(Module):
    sync = True
    checking_mode = False

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 device_ids=None):
        super(BatchnormSync, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        self.mean = torch.zeros(num_features)
        self.std = torch.ones(num_features)

        self.reset_parameters()
        self.cum_queue = Queue()
        self.broadcast_queue = Queue()
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = device_ids

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.mean.zero_()
        self.std.fill_(1)
        if self.affine:
            if BatchnormSync.checking_mode:
                self.weight.data.fill_(1)
            else:
                self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, input, input_lengths):
        training = int(self.training)
        assert input.size(1) == self.num_features

        bn_func = BatchnormFunction(
            self.running_mean, self.running_var,
            training, self.cum_queue, self.broadcast_queue, self.device_ids,
            BatchnormSync.sync, self.eps, self.momentum, self.affine)
        return bn_func(input, input_lengths, self.weight, self.bias)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))
