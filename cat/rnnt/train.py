# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""
Transducer trainer.
"""

__all__ = ["TransducerTrainer", "build_model", "_parser", "main"]

from . import joiner as joiner_zoo
from ..shared import (
    SpecAug,
    Manager
)
from ..shared import coreutils
from ..shared import encoder as tn_zoo
from ..shared import decoder as pn_zoo
from ..shared.layer import SampledSoftmax
from ..shared.data import (
    KaldiSpeechDataset,
    sortedPadCollateASR
)

import os
import gather
import argparse
from collections import OrderedDict
from warp_rnnt import rnnt_loss as RNNTLoss
from typing import *

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace, **mkwargs):
    coreutils.set_random_seed(args.seed)
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if 'Dataset' not in mkwargs:
        mkwargs['Dataset'] = KaldiSpeechDataset

    if 'collate_fn' not in mkwargs:
        mkwargs['collate_fn'] = sortedPadCollateASR()

    if 'func_build_model' not in mkwargs:
        mkwargs['func_build_model'] = build_model

    mkwargs['args'] = args
    manager = Manager(**mkwargs)

    # training
    manager.run(args)


class TransducerTrainer(nn.Module):
    # fmt: off
    def __init__(
        self,
        encoder: tn_zoo.AbsEncoder = None,
        predictor: pn_zoo.AbsDecoder = None,
        joiner: joiner_zoo.AbsJointNet = None,
        # enable compact layout, would consume less memory and fasten the computation of RNN-T loss.
        compact: bool = False,
        # weight of ILME loss to conduct joiner-training
        ilme_weight: Optional[float] = None,
        # add mask to predictor output, this specifies the range of mask
        predictor_mask_range: float = 0.1,
        # add mask to predictor output, this specifies the # mask
        num_predictor_mask: int = -1,
        bos_id: int = 0,
        # conduct sampled softmax
        sampled_softmax: bool = False,
        sampled_softmax_uniform_ratio: float = 0.0):
        # fmt: on
        super().__init__()

        if sampled_softmax:
            assert bos_id == 0
            assert joiner.is_normalize_separated

        self.ilme_weight = 0.0
        if ilme_weight is not None and ilme_weight != 0.0:
            if not isinstance(joiner, joiner_zoo.JointNet):
                raise NotImplementedError(
                    f"TransducerTrainer: \n"
                    f"ILME loss joiner training only support joiner network 'JointNet', instead of {joiner.__class__.__name__}")
            self._ilme_criterion = nn.CrossEntropyLoss(reduction='sum')
            self.ilme_weight = ilme_weight

        self._compact = compact
        self._sampled_softmax = sampled_softmax
        if sampled_softmax:
            self.log_softmax = SampledSoftmax(
                blank=0, uniform_ratio=sampled_softmax_uniform_ratio)

        self.encoder = encoder
        self.predictor = predictor
        self.joiner = joiner

        if compact:
            if not hasattr(self.joiner, 'iscompact'):
                print(
                    "warning: it seems the joiner network might not be compatible with 'compact=Ture'.")
            else:
                assert self.joiner.iscompact == compact

        if num_predictor_mask != -1:
            self._pn_mask = SpecAug(
                time_mask_width_range=predictor_mask_range,
                num_time_mask=num_predictor_mask,
                apply_freq_mask=False,
                apply_time_warp=False
            )
        else:
            self._pn_mask = None

        self.bos_id = bos_id

    def compute_join(self, enc_out: torch.Tensor, pred_out: torch.Tensor, targets: torch.Tensor, enc_out_lens: torch.Tensor, target_lens: torch.Tensor) -> torch.FloatTensor:
        device = enc_out.device
        enc_out_lens = enc_out_lens.to(device=device, dtype=torch.int)
        targets = targets.to(device=device, dtype=torch.int)
        target_lens = target_lens.to(device=device, dtype=torch.int)

        if self._pn_mask is not None:
            pred_out = self._pn_mask(pred_out, target_lens+1)[0]

        if self._compact:
            # squeeze targets to 1-dim
            targets = targets.to(enc_out.device, non_blocking=True)
            if targets.dim() == 2:
                targets = gather.cat(targets, target_lens)

        if self._sampled_softmax:
            logits = self.joiner.impl_forward(
                enc_out, pred_out, enc_out_lens, target_lens+1
            )
            joinout, targets = self.log_softmax(
                logits, targets
            )
        else:
            joinout = self.joiner(
                enc_out, pred_out, enc_out_lens, target_lens+1)

        return joinout, targets, enc_out_lens, target_lens

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, in_lens: torch.LongTensor, target_lens: torch.LongTensor) -> torch.FloatTensor:

        enc_out, enc_out_lens = self.encoder(inputs, in_lens)
        pred_out = self.predictor(torch.nn.functional.pad(
            targets, (1, 0), value=self.bos_id))[0]

        joinout, targets, enc_out_lens, target_lens = self.compute_join(
            enc_out, pred_out, targets, enc_out_lens, target_lens
        )
        loss = 0.0
        if self.ilme_weight != 0.0:
            # calculate the ILME ILM loss
            # ilm_log_probs: (N, 1, V) + (N, Up-1, V) -> (N, U, V)
            ilm_log_probs = self.joiner(
                pred_out.new_zeros(pred_out.size(0), 1, enc_out.size(-1)),
                pred_out[:, :-1, :]).squeeze(1)
            # ilm_log_probs: (N, U, V) -> (\sum(U_i), V)
            ilm_log_probs = gather.cat(ilm_log_probs, target_lens)
            if targets.dim() == 2:
                # normal layout -> compact layout
                # ilm_targets: (\sum{U_i}, )
                ilm_targets = gather.cat(targets, target_lens)
            elif targets.dim() == 1:
                ilm_targets = targets
            else:
                raise ValueError(
                    f"{self.__class__.__name__}: invalid dimension of targets '{targets.dim()}', expected 1 or 2.")

            loss += self.ilme_weight / enc_out.size(0) * \
                self._ilme_criterion(ilm_log_probs, ilm_targets)

        loss += RNNTLoss(
            joinout, targets,
            enc_out_lens,
            target_lens,
            reduction='mean', gather=True, compact=self._compact
        )

        return loss


@torch.no_grad()
def build_model(
        cfg: dict,
        args: Optional[Union[argparse.Namespace, dict]] = None,
        dist: bool = True,
        wrapped: bool = True) -> Union[nn.parallel.DistributedDataParallel, TransducerTrainer, Tuple[tn_zoo.AbsEncoder, pn_zoo.AbsDecoder, joiner_zoo.AbsJointNet]]:
    """
    cfg:
        trainer:
            please refer to TransducerTrainer.__init__() for support arguments
        joiner:
            ...
        encoder:
            ...
        decoder:
            ...
    """
    if args is not None:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        elif not isinstance(args, dict):
            raise ValueError(f"unsupport type of args: {type(args)}")

    def _load_and_immigrate(orin_dict_path: str, str_src: str, str_dst: str) -> OrderedDict:
        if not os.path.isfile(orin_dict_path):
            raise FileNotFoundError(f"{orin_dict_path} is not a valid file.")

        checkpoint = torch.load(orin_dict_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            new_state_dict[k.replace(str_src, str_dst)] = v
        del checkpoint
        return new_state_dict

    def _build(c_cfg: dict, component: Literal['encoder', 'predictor', 'joiner']) -> Union[tn_zoo.AbsEncoder, pn_zoo.AbsDecoder, joiner_zoo.AbsJointNet]:
        assert 'kwargs' in c_cfg

        if component == 'encoder':
            zoo = tn_zoo
        elif component == 'predictor':
            zoo = pn_zoo
        elif component == 'joiner':
            zoo = joiner_zoo
        else:
            raise ValueError(f"Unknow component: {component}")

        _model = getattr(zoo, c_cfg['type'])(**c_cfg['kwargs'])

        if "pretrained" in c_cfg:
            if component == "encoder":
                prefix = 'module.am.'
            elif component == "predictor":
                prefix = 'module.lm.'
            else:
                raise RuntimeError(
                    "Unsupport component with 'pretrained' option: {}".format(component))

            _model.load_state_dict(
                _load_and_immigrate(c_cfg['pretrained'], prefix, ''), strict=False)

        if c_cfg.get('freeze', False):
            _model.requires_grad_(False)
        return _model

    assert 'encoder' in cfg
    assert 'decoder' in cfg
    assert 'joiner' in cfg

    encoder = _build(cfg['encoder'], 'encoder')
    predictor = _build(cfg['decoder'], 'predictor')
    joiner = _build(cfg['joiner'], 'joiner')

    if not wrapped:
        return encoder, predictor, joiner

    # for compatible of old settings
    transducer_kwargs = cfg.get('trainer', {})

    model = TransducerTrainer(
        encoder=encoder,
        predictor=predictor,
        joiner=joiner,
        **transducer_kwargs
    )

    if not dist:
        return model

    assert args is not None, f"You must tell the GPU id to build a DDP model."

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)

    model.cuda(args['gpu'])
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args['gpu']])
    return model


def _parser():
    return coreutils.basic_trainer_parser("Transducer training")


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)


if __name__ == "__main__":
    print(
        "NOTE:\n"
        "    since we import the build_model() function in cat.rnnt,\n"
        "    we should avoid calling `python -m cat.rnnt.train`, instead\n"
        "    running `python -m cat.rnnt`"
    )
