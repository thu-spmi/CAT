# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""
Transducer trainer.
"""

__all__ = ["TransducerTrainer", "build_model", "_parser", "main"]

from . import joiner as joiner_zoo
from ..shared import SpecAug, Manager
from ..shared import coreutils
from ..shared import encoder as tn_zoo
from ..shared import decoder as pn_zoo
from ..shared.data import KaldiSpeechDataset, sortedPadCollateASR

import os
import math
import gather
import argparse
import warp_rnnt

try:
    import warp_ctct
except ModuleNotFoundError:
    print(
        "WARNING: 'warp-ctct' not installed. Refer to https://github.com/maxwellzh/warp-ctct"
    )
from collections import OrderedDict
from typing import *

import torch
import torch.nn as nn
import torch.distributed as dist


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace, **mkwargs):
    coreutils.set_random_seed(args.seed)
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    if "T_dataset" not in mkwargs:
        mkwargs["T_dataset"] = KaldiSpeechDataset

    if "collate_fn" not in mkwargs:
        mkwargs["collate_fn"] = sortedPadCollateASR()

    if "func_build_model" not in mkwargs:
        mkwargs["func_build_model"] = build_model

    mkwargs["args"] = args
    manager = Manager(**mkwargs)

    # training
    manager.run(args)


class SeqExtractor(nn.Module):
    """Return a tuple of sequences (<bos>+y, y+<eos>) for given y

    Examples:
        >>> # (N, 1+U), (N, U+1), (N, U), (N, )
        >>> # ly is optional
        >>> ypad, target = seqextractor(y, ly)
    """

    def __init__(self, bos: int, eos: int) -> None:
        super().__init__()
        self.pad_bos = nn.ConstantPad1d((1, 0), bos)
        self.pad_zero = nn.ConstantPad1d((0, 1), 0)
        self.register_buffer("eos", torch.tensor(eos).view(1, 1), persistent=False)

    def forward(
        self,
        y: torch.Tensor,
        ly: Optional[torch.Tensor] = None,
        return_pad: bool = True,
        return_target: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out = []
        if return_pad:
            ypad = self.pad_bos(y)
            out.append(ypad)

        if return_target:
            target = self.pad_zero(y)
            if ly is None:
                target[:, -1] = self.eos[0]
            else:
                target.scatter_(
                    dim=1,
                    index=ly.unsqueeze(1).long(),
                    src=self.eos.expand(target.shape),
                )
            out.append(target)

        if len(out) == 1:
            return out[0]
        return tuple(out)


class TransducerTrainer(nn.Module):
    def __init__(
        self,
        encoder: tn_zoo.AbsEncoder = None,
        predictor: pn_zoo.AbsDecoder = None,
        joiner: joiner_zoo.AbsJointNet = None,
        # enable compact layout, would consume less memory and fasten the computation of RNN-T loss.
        compact: bool = False,
        # add mask to predictor output, this specifies the range of mask
        predictor_mask_range: float = 0.1,
        # add mask to predictor output, this specifies the # mask
        num_predictor_mask: int = -1,
        # <eos> id, optional, if -1, <eos> won't be considered.
        eos_id: int = -1,
        topo: Literal["rnnt", "ctct"] = "rnnt",
    ):
        super().__init__()
        assert isinstance(compact, bool)
        assert isinstance(predictor_mask_range, (int, float))
        assert isinstance(num_predictor_mask, int)
        assert isinstance(eos_id, int)

        assert topo in ("rnnt", "ctct")
        self.topo = topo
        self._compact = compact

        self.encoder = encoder
        self.predictor = predictor
        self.joiner = joiner

        if compact:
            if not hasattr(self.joiner, "iscompact"):
                print(
                    "WARNING: it seems the joiner network might not be compatible with 'compact=Ture'."
                )
            else:
                assert self.joiner.iscompact == compact

        if num_predictor_mask != -1:
            self._pn_mask = SpecAug(
                time_mask_width_range=predictor_mask_range,
                num_time_mask=num_predictor_mask,
                apply_freq_mask=False,
                apply_time_warp=False,
            )
        else:
            self._pn_mask = None

        assert eos_id >= -1
        self.append_eos = eos_id != -1
        self.eos_id = eos_id
        self.seq_extractor = SeqExtractor(bos=0, eos=(eos_id if eos_id > 0 else 0))

    def compute_join(
        self,
        enc_out: torch.Tensor,
        pred_out: torch.Tensor,
        y: torch.Tensor,
        lsub: torch.Tensor,
        ly: torch.Tensor,
    ) -> torch.FloatTensor:
        device = enc_out.device
        lsub = lsub.to(device=device, dtype=torch.int)
        y = y.to(device=device, dtype=torch.int)
        ly = ly.to(device=device, dtype=torch.int)

        if self._compact:
            # squeeze targets to 1-dim
            y = y.to(enc_out.device, non_blocking=True)
            if y.dim() == 2:
                y = gather.cat(y, ly)

        xys = self.joiner(enc_out, pred_out, lsub, ly + 1)

        return xys, y, lsub, ly

    def forward(
        self,
        x: torch.FloatTensor,
        lx: torch.LongTensor,
        y: torch.LongTensor,
        ly: torch.LongTensor,
    ) -> torch.FloatTensor:
        enc_out, lsub = self.encoder(x, lx)

        if self.append_eos:
            y = self.seq_extractor(y, ly, return_pad=False)
            ly += 1

        pred_out = self.predictor(self.seq_extractor(y, return_target=False))[0]

        if self._pn_mask is not None:
            pred_out = self._pn_mask(pred_out, ly + 1)[0]

        if isinstance(self.joiner, joiner_zoo.LogAdd):
            if self.topo == "rnnt":
                fn = warp_rnnt.rnnt_loss_simple
            elif self.topo == "ctct":
                fn = warp_ctct.ctct_simple_loss
            else:
                raise NotImplementedError
            return fn(enc_out, pred_out, y, lsub, ly)

        xys, y, lsub, ly = self.compute_join(enc_out, pred_out, y, lsub, ly)
        if self.topo == "rnnt":
            loss = warp_rnnt.rnnt_loss(xys, y, lsub, ly, compact=self._compact)
        elif self.topo == "ctct":
            loss = warp_ctct.ctct_loss(xys, y, lsub, ly)
        else:
            raise NotImplementedError(f"unknown type of topo: {self.topo}")

        return loss


@torch.no_grad()
def build_model(
    cfg: dict,
    args: Optional[Union[argparse.Namespace, dict]] = None,
    dist: bool = True,
    wrapped: bool = True,
) -> Union[
    nn.parallel.DistributedDataParallel,
    TransducerTrainer,
    Tuple[tn_zoo.AbsEncoder, pn_zoo.AbsDecoder, joiner_zoo.AbsJointNet],
]:
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

    def _load_and_immigrate(
        orin_dict_path: str, str_src: str, str_dst: str
    ) -> OrderedDict:
        if not os.path.isfile(orin_dict_path):
            raise FileNotFoundError(f"{orin_dict_path} is not a valid file.")

        checkpoint = torch.load(orin_dict_path, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            new_state_dict[k.replace(str_src, str_dst)] = v
        del checkpoint
        return new_state_dict

    def _build(
        c_cfg: dict, component: Literal["encoder", "predictor", "joiner"]
    ) -> Union[tn_zoo.AbsEncoder, pn_zoo.AbsDecoder, joiner_zoo.AbsJointNet]:
        assert "kwargs" in c_cfg

        if component == "encoder":
            zoo = tn_zoo
        elif component == "predictor":
            zoo = pn_zoo
        elif component == "joiner":
            zoo = joiner_zoo
        else:
            raise ValueError(f"Unknow component: {component}")

        _model = getattr(zoo, c_cfg["type"])(**c_cfg["kwargs"])

        if "init" in c_cfg:
            if component == "encoder":
                prefix = "module.encoder."
            elif component == "predictor":
                prefix = "module.lm."
            else:
                raise RuntimeError(
                    "Unsupport component with 'init' option: {}".format(component)
                )
            # NOTE (huahuan):
            # When init parts of the transducer, `strict=False` is set,
            # it's your duty to ensure the keys are mapped in the state dict.
            _model.load_state_dict(
                _load_and_immigrate(c_cfg["init"], prefix, ""), strict=False
            )

        if c_cfg.get("freeze", False):
            _model.requires_grad_(False)
        return _model

    assert "encoder" in cfg
    assert "decoder" in cfg
    assert "joiner" in cfg

    encoder = _build(cfg["encoder"], "encoder")
    predictor = _build(cfg["decoder"], "predictor")
    joiner = _build(cfg["joiner"], "joiner")

    if not wrapped:
        return encoder, predictor, joiner

    # for compatible of old settings
    transducer_kwargs = cfg.get("trainer", {})

    model = TransducerTrainer(
        encoder=encoder, predictor=predictor, joiner=joiner, **transducer_kwargs
    )

    if not dist:
        return model

    assert args is not None, f"You must tell the GPU id to build a DDP model."

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)

    model.cuda(args["gpu"])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args["gpu"]])
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
