# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Keyu An, Huahuan Zheng

__all__ = ["UnifiedAMTrainer", "build_model", "_parser", "main"]

from .train_me2e import AMTrainer, build_model as am_builder, main_worker as basic_worker
from ..shared import coreutils
from ..shared.simu_net import SimuNet

import os
import argparse
from typing import *

import torch
import torch.nn as nn
import random
import math
import numpy as np
from torch.cuda.amp import autocast




def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    basic_worker(gpu, ngpus_per_node, args, func_build_model=build_model)


class UnifiedAMTrainer(AMTrainer):
    def __init__(
        self,
        # chunk related parameters
        # configure according to the encoder
        downsampling_ratio: int = 4,
        chunk_size: int = 40,
        context_size_left: int = 40,
        context_size_right: int = 40,
        # jitter is applied after the downsampling
        jitter_range: int = 2,
        mel_dim: int = 80,
        simu: bool = False,
        simu_loss_weight: float = 1.0,

        **kwargs,
    ):
        super().__init__(**kwargs)


        self.simu = simu
        if self.simu:
            self.simu_net = SimuNet(
                mel_dim=mel_dim, out_len=context_size_right, hdim=256, rnn_num_layers=3
            )
            self.simu_loss = nn.L1Loss()
        
        self.simu_loss_weight = simu_loss_weight
        self.chunk_size = chunk_size
        self.context_size_left = context_size_left
        self.context_size_right = context_size_right
        self.jitter_range = jitter_range
        self.downsampling_ratio = downsampling_ratio
        

    def bf_chunk_infer(
        self, inputs: torch.FloatTensor, in_lens: torch.LongTensor
    ) -> torch.FloatTensor:
        
        # time domain to fre domain
        inputs, flens = self.stft(inputs, in_lens)
        
        chunk_size = self.chunk_size
        
        max_input_length = int(
            chunk_size * (math.ceil(float(inputs.shape[1]) / chunk_size))
        )
        inputs = map(lambda x: pad_to_len(x, max_input_length, 0), inputs)
        inputs = list(inputs)
        inputs = torch.stack(inputs, dim=0)

        left_context_size = self.context_size_left

        num_chunks = inputs.size(1) // chunk_size
        
        assert num_chunks > 0
        
        BC = inputs.size(0) * num_chunks
        Channel = inputs.size(2)
        Fre = inputs.size(3)
        RI = inputs.size(4)
        inputs = inputs.view(BC, chunk_size, Channel,Fre,RI)

        left_context = torch.zeros(
            BC, left_context_size, Channel, Fre,RI,device=inputs.device
            )

        # fill first left chunk with zeros
        if left_context_size > chunk_size:
            N = left_context_size // chunk_size
            for idx in range(N):
                left_context[
                    N - idx :, idx * chunk_size : (idx + 1) * chunk_size, ...
                ] = inputs[: -N + idx, :, ...]
            for idx in range(N):
                left_context[idx::num_chunks, : (N - idx) * chunk_size, ...] = 0
        else:
            left_context[1:, :, ...] = inputs[:-1, -left_context_size:, ...]
            left_context[0::num_chunks, :, ...] = 0

        if self.context_size_right > 0:
            right_context = torch.zeros(
                BC, self.context_size_right, Channel, Fre,RI, device=inputs.device
            )
            if self.context_size_right > chunk_size:
                right_context[:-1, :chunk_size, ...] = inputs[1:, :, ...]
                right_context[:-2, chunk_size:, ...] = inputs[
                    2:, : self.context_size_right - chunk_size, ...
                ]
                right_context[num_chunks - 1 :: num_chunks, :, ...] = 0
                right_context[num_chunks - 2 :: num_chunks, chunk_size:, ...] = 0
            else:
                right_context[:-1, :, ...] = inputs[1:, : self.context_size_right, ...]
                right_context[num_chunks - 1 :: num_chunks, :, ...] = 0

            if self.simu:
                if self.training :
                    if np.random.rand() < 0.5:
                        contexted_inputs = (left_context, inputs)
                    else:
                        contexted_inputs = (left_context, inputs, right_context)
                else:
                    contexted_inputs = (left_context, inputs)
            else:
                #simu_loss = 0
                if self.training and np.random.rand() < 0.5:
                    contexted_inputs = (left_context, inputs,right_context)
                else:
                    contexted_inputs = (left_context, inputs)
            
            inputs_with_context = torch.cat(contexted_inputs, dim=1)
        else:
            inputs_with_context = torch.cat((left_context, inputs), dim=1)
        
        flens1 = torch.full([inputs_with_context.size(0)], inputs_with_context.size(1))
            
        
        # front end
        flens_chunk = torch.full([inputs_with_context.size(0)], inputs_with_context.size(1))
            
        inputs_with_context, flens1,_ = self.beamformer(
            inputs_with_context,
            flens_chunk,
            )
        assert (flens_chunk == flens1).all()
        
        if torch.isnan(inputs_with_context).any() or torch.isinf(inputs_with_context).any():
            raise ValueError("feats array contains NaN or Infinity values!")
        
        # feature exaction
        input_power = inputs_with_context[..., 0] ** 2 + inputs_with_context[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        stream_chunk, _ = self.logmel(input_amp, flens_chunk)    
        
        if self.simu:
            # FIXME: maybe .clone() is not required
            left_context = stream_chunk[:, :left_context_size, :]
            inputs_without_context = stream_chunk[:, left_context_size:chunk_size + left_context_size, :]
            simu_right_context = self.simu_net(inputs_without_context, chunk_size)

            mel_dim = stream_chunk.size(2)
            
            if self.context_size_right > 0:
                right_context = torch.zeros(
                    BC, self.context_size_right, mel_dim, device=inputs_without_context.device
                )
                if self.context_size_right > chunk_size:
                    right_context[:-1, :chunk_size, :] = inputs_without_context[1:, :, :]
                    right_context[:-2, chunk_size:, :] = inputs_without_context[
                        2:, : self.context_size_right - chunk_size, :
                    ]
                    right_context[num_chunks - 1 :: num_chunks, :, :] = 0
                    right_context[num_chunks - 2 :: num_chunks, chunk_size:, :] = 0
                else:
                    right_context[:-1, :, :] = inputs_without_context[1:, : self.context_size_right, :]
                    right_context[num_chunks - 1 :: num_chunks, :, :] = 0

                
                simu_loss = self.simu_loss(simu_right_context, right_context.detach())
                if self.training:
                    if np.random.rand() < 0.5:
                        contexted_inputs = (left_context, inputs_without_context, simu_right_context)
                    elif np.random.rand() < 0.5:
                        contexted_inputs = (left_context, inputs_without_context)
                    else:
                        contexted_inputs = (left_context, inputs_without_context, right_context)
                else:
                    contexted_inputs = (left_context, inputs_without_context, simu_right_context)
                
                stream_chunk = torch.cat(contexted_inputs, dim=1)
            else:
                stream_chunk = torch.cat((left_context, inputs_without_context), dim=1)    
            
            
        # ASR    
        enc_out_with_context, _ = self.encoder(
            stream_chunk,
            flens1,
        )
        enc_out = enc_out_with_context[
            :,
            left_context_size
            // self.downsampling_ratio : (chunk_size + left_context_size)
            // self.downsampling_ratio,
            :,
        ]
        enc_out = enc_out.contiguous().view(
            enc_out.size(0) // num_chunks, enc_out.size(1) * num_chunks, -1
        )

        out_lens = torch.div(
            chunk_size * torch.ceil(flens / chunk_size),
            self.downsampling_ratio,
            rounding_mode="floor",
        )
        return enc_out, out_lens

    def forward(self, audio, lx, labels, ly):
        # Chunk divide
        
        jitter = self.downsampling_ratio * random.randint(
            -self.jitter_range, self.jitter_range
        )
        chunk_size = self.chunk_size + jitter
        
        inputs, flens = self.stft(audio, lx)
        
        
        max_input_length = int(
            chunk_size * (math.ceil(float(inputs.shape[1]) / chunk_size))
        )
        
        inputs = map(lambda x: pad_to_len(x, max_input_length, 0), inputs)
        inputs = list(inputs)
        inputs = torch.stack(inputs, dim=0)

        num_chunks = inputs.size(1) // chunk_size
        
        assert num_chunks > 0
        
        BC = inputs.size(0) * num_chunks
        Channel = inputs.size(2)
        Fre = inputs.size(3)
        RI = inputs.size(4)
        inputs = inputs.view(BC, chunk_size, Channel,Fre,RI)
        

        # setup left context
        left_context_size = self.context_size_left + jitter * (
            self.context_size_left // self.chunk_size
        )
        left_context = torch.zeros(BC, left_context_size, Channel, Fre,RI,device=inputs.device)
        # fill first left chunk with zeros
        if left_context_size > chunk_size:
            N = left_context_size // chunk_size
            for idx in range(N):
                left_context[
                    N - idx :, idx * chunk_size : (idx + 1) * chunk_size, ...
                ] = inputs[: -N + idx, :, ...]
            for idx in range(N):
                left_context[idx::num_chunks, : (N - idx) * chunk_size, ...] = 0
        else:
            left_context[1:, :, ...] = inputs[:-1, -left_context_size:, ...]
            left_context[0::num_chunks, :, ...] = 0

        if self.context_size_right > 0:
            right_context = torch.zeros(
                BC, self.context_size_right, Channel, Fre,RI, device=inputs.device
            )
            if self.context_size_right > chunk_size:
                right_context[:-1, :chunk_size, ...] = inputs[1:, :, ...]
                right_context[:-2, chunk_size:, ...] = inputs[
                    2:, : self.context_size_right - chunk_size, ...
                ]
                right_context[num_chunks - 1 :: num_chunks, :, ...] = 0
                right_context[num_chunks - 2 :: num_chunks, chunk_size:, ...] = 0
            else:
                right_context[:-1, :, ...] = inputs[1:, : self.context_size_right, ...]
                right_context[num_chunks - 1 :: num_chunks, :, ...] = 0

            if self.simu:
                if self.training :
                    if np.random.rand() < 0.5:
                        contexted_inputs = (left_context, inputs)
                    else:
                        contexted_inputs = (left_context, inputs, right_context)
                else:
                    contexted_inputs = (left_context, inputs)
            else:
                simu_loss = 0
                if self.training and np.random.rand() < 0.5:
                    contexted_inputs = (left_context, inputs)
                else:
                    contexted_inputs = (left_context, inputs, right_context)
            
            inputs_with_context = torch.cat(contexted_inputs, dim=1)
        else:
            inputs_with_context = torch.cat((left_context, inputs), dim=1)
        
        flens1 = torch.full([inputs_with_context.size(0)], inputs_with_context.size(1))
        
        # Stream Front end 
        inputs_with_context, flens1,_ = self.beamformer(
                inputs_with_context,
                flens1,
                )

        
        if torch.isnan(inputs_with_context).any() or torch.isinf(inputs_with_context).any():
            raise ValueError("feats array contains NaN or Infinity values!")

        
        # utt cat 
        samples = inputs_with_context[:, left_context_size:chunk_size + left_context_size, :]
        samples = samples.contiguous().view(samples.size(0)//num_chunks, 
                                            samples.size(1)*num_chunks, 
                                            samples.size(2), 
                                            samples.size(3))
            
        
        # utt feature exaction
        input_power = samples[..., 0] ** 2 + samples[..., 1] ** 2 
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))

        samples, _ = self.logmel(input_amp, flens)
        
        # utt ASR
        logits, lx = self.encoder(samples, flens)
        logits = torch.log_softmax(logits, dim=-1)
        
        # utt loss
        labels = labels.cpu()
        lx = lx.cpu()
        ly = ly.cpu()
        if self.is_crf:
            if self._crf_ctx is None:
                # lazy init
                self.register_crf_ctx(self.den_lm)
            with autocast(enabled=False):
                loss = self.criterion(
                    logits.float(),
                    labels.to(torch.int),
                    lx.to(torch.int),
                    ly.to(torch.int),
                )
        else:
            # [N, T, C] -> [T, N, C]
            logits = logits.transpose(0, 1)
            loss = self.criterion(
                logits, labels.to(torch.int), lx.to(torch.int), ly.to(torch.int)
            )


        # chunk feature exaction
        input_power = inputs_with_context[..., 0] ** 2 + inputs_with_context[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        stream_chunk, _ = self.logmel(input_amp, flens1)
        
        # Chunk simu or not
        if self.simu:
            # FIXME: maybe .clone() is not required
            left_context = stream_chunk[:, :left_context_size, :]
            inputs_without_context = stream_chunk[:, left_context_size:chunk_size + left_context_size, :]
            simu_right_context = self.simu_net(inputs_without_context, chunk_size)

            mel_dim = stream_chunk.size(2)
            
            if self.context_size_right > 0:
                right_context = torch.zeros(
                    BC, self.context_size_right, mel_dim, device=inputs_without_context.device
                )
                if self.context_size_right > chunk_size:
                    right_context[:-1, :chunk_size, :] = inputs_without_context[1:, :, :]
                    right_context[:-2, chunk_size:, :] = inputs_without_context[
                        2:, : self.context_size_right - chunk_size, :
                    ]
                    right_context[num_chunks - 1 :: num_chunks, :, :] = 0
                    right_context[num_chunks - 2 :: num_chunks, chunk_size:, :] = 0
                else:
                    right_context[:-1, :, :] = inputs_without_context[1:, : self.context_size_right, :]
                    right_context[num_chunks - 1 :: num_chunks, :, :] = 0

                
                simu_loss = self.simu_loss(simu_right_context, right_context.detach())
                if self.training:
                    if np.random.rand() < 0.5:
                        contexted_inputs = (left_context, inputs_without_context, simu_right_context)
                    elif np.random.rand() < 0.5:
                        contexted_inputs = (left_context, inputs_without_context)
                    else:
                        contexted_inputs = (left_context, inputs_without_context, right_context)
                else:
                    contexted_inputs = (left_context, inputs_without_context, simu_right_context)
                
                stream_chunk = torch.cat(contexted_inputs, dim=1)
            else:
                stream_chunk = torch.cat((left_context, inputs_without_context), dim=1)    
            
            flens1 = torch.full([stream_chunk.size(0)], stream_chunk.size(1))
            
        
        # Chunk ASR
        enc_out_with_context, _ = self.encoder(
            stream_chunk,
            flens1,
        )
        enc_out = enc_out_with_context[
            :,
            left_context_size
            // self.downsampling_ratio : (chunk_size + left_context_size)
            // self.downsampling_ratio,
            :,
        ]
        enc_out = enc_out.contiguous().view(
            enc_out.size(0) // num_chunks, enc_out.size(1) * num_chunks, -1
        )
        
        
        enc_out = enc_out[:, : lx[0].int(), :]
        chunk_logits = torch.log_softmax(enc_out, dim=-1)

        # Chunk loss
        if self.is_crf:
            with autocast(enabled=False):
                chunk_loss = self.criterion(
                    chunk_logits.float(),
                    labels.to(torch.int),
                    lx.to(torch.int),
                    ly.to(torch.int),
                )
        else:
            # [N, T, C] -> [T, N, C]
            chunk_logits = chunk_logits.transpose(0, 1)
            chunk_loss = self.criterion(
                chunk_logits, labels.to(torch.int), lx.to(torch.int), ly.to(torch.int)
            )

        
        
        if float('inf') == loss + chunk_loss + simu_loss :
            raise ValueError("loss contains Infinity values!")
        
        if math.isnan(loss + chunk_loss + simu_loss ):
            raise ValueError("loss contains NaN values!")

        return loss + chunk_loss + (simu_loss * self.simu_loss_weight), loss, chunk_loss, simu_loss


def pad_to_len(t: torch.Tensor, pad_len: int, dim: int):
    """Pad the tensor `t` at `dim` to the length `pad_len` with right padding zeros."""
    if t.size(dim) == pad_len:
        return t
    else:
        pad_size = list(t.shape)
        pad_size[dim] = pad_len - t.size(dim)
        return torch.cat(
            [t, torch.zeros(*pad_size, dtype=t.dtype, device=t.device)], dim=dim
        )


def build_model(
    cfg: dict, args: Optional[argparse.Namespace] = None, dist: bool = True
):
    """
    cfg: refer to UnifiedAMTrainer.__init__()
    """
    assert "trainer" in cfg, f"missing 'trainer' in field:"
    cfg["trainer"]["encoder"] = am_builder(cfg, args, dist=False, wrapper=False)
    model = UnifiedAMTrainer(**cfg["trainer"])
    if not dist:
        return model

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        #find_unused_parameters=True, 
        device_ids=[args.gpu])

    return model


def _parser():
    return coreutils.basic_trainer_parser("Unified streaming/offline CTC/CRF Trainer")


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)


if __name__ == "__main__":
    main()
