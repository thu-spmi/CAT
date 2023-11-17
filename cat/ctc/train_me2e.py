# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""Top interface of CTC training.
"""

__all__ = ["AMTrainer", "build_model", "_parser", "main"]

import torchaudio
from ..shared import Manager
from ..shared import coreutils
from ..shared import encoder as model_zoo
from ..shared.data import KaldiSpeechDataset, sortedPadCollateASR

from ..front.stft import Stft
from ..front.log_mel import LogMel
from ..front.beamformer_net import BeamformerNet

import os
import argparse
import Levenshtein
from typing import *
from ctcdecode import CTCBeamDecoder
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast

import math
import numpy as np

# NOTE (huahuan):
#   1/4 subsampling is used for Conformer model defaultly
#   for other sampling ratios, you need to modify the value.
#   Commonly, you can use a relatively larger value for allowing some margin.
SUBSAMPLING = 4


def check_label_len_for_ctc(
    tupled_mat_label: Tuple[torch.FloatTensor, torch.LongTensor]
):
    """filter the short seqs for CTC/CRF"""
    return tupled_mat_label[0].shape[0] // SUBSAMPLING > tupled_mat_label[1].shape[0]


def filter_hook(dataset):
    return dataset.select(check_label_len_for_ctc)


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
        mkwargs["collate_fn"] = sortedPadCollateASR(flatten_target=True)

    if "func_build_model" not in mkwargs:
        mkwargs["func_build_model"] = build_model

    if "_wds_hook" not in mkwargs:
        mkwargs["_wds_hook"] = filter_hook

    if (
        "func_eval" not in mkwargs
        and hasattr(args, "eval_error_rate")
        and args.eval_error_rate
    ):
        mkwargs["func_eval"] = custom_evaluate

    mkwargs["args"] = args
    manager = Manager(**mkwargs)

    # NOTE: for CTC training, the input feat len must be longer than the label len
    #       ... when using webdataset (--ld) to load the data, we deal with
    #       ... the issue by `_wds_hook`; if not, we filter the unqualified utterances
    #       ... before training start.
    if args.ld is None:
        coreutils.distprint(f"> filter seqs by ctc restriction", args.gpu)
        tr_dataset = manager.trainloader.dl.dataset
        orilen = len(tr_dataset)
        tr_dataset.filt_by_len(lambda x, y: x // SUBSAMPLING > y)
        coreutils.distprint(
            f"  filtered {orilen-len(tr_dataset)} utterances.", args.gpu
        )

    # training
    manager.run(args)


class AMTrainer(nn.Module):
    def __init__(
        self,
        encoder: model_zoo.AbsEncoder,
        use_crf: bool = False,
        den_lm: Optional[str] = None,
        lamb: Optional[float] = 0.01,
        decoder: CTCBeamDecoder = None,
        # front
        n_fft: int = 512,
        win_length:int = 400,
        hop_length:int = 160,
        idim: int = 80,
        beamforming: str = "mvdr",
        wpe: bool = False,
        ref_ch: int = 0,
    ):
        super().__init__()

        self.stft = Stft(n_fft, win_length, hop_length)
        self.logmel = LogMel(n_mels=idim)
        
        if wpe:
            self.beamformer = BeamformerNet(beamformer_type=beamforming,use_wpe=True,
                                            use_dnn_mask_for_wpe=True, ref_channel=ref_ch)
        else:
            self.beamformer = BeamformerNet(beamformer_type=beamforming,ref_channel=ref_ch)
        
        self.encoder = encoder
        self.is_crf = use_crf
        if use_crf:
            self.den_lm = den_lm
            assert den_lm is not None and os.path.isfile(den_lm)

            from ctc_crf import CTC_CRF_LOSS as CRFLoss

            self.criterion = CRFLoss(lamb=lamb)
            self._crf_ctx = None
        else:
            self.den_lm = None
            self.criterion = nn.CTCLoss()
        
        self.attach = {"decoder": decoder}

    def clean_unpickable_objs(self):
        # CTCBeamDecoder is unpickable,
        # So, this is required for inference.
        self.attach["decoder"] = None

    def register_crf_ctx(self, den_lm: Optional[str] = None):
        """Register the CRF context on model device."""
        assert self.is_crf

        from ctc_crf import CRFContext

        self._crf_ctx = CRFContext(
            den_lm, next(iter(self.encoder.parameters())).device.index
        )

    @torch.no_grad()
    def get_wer(
        self, xs: torch.Tensor, ys: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor
    ):
        if self.attach["decoder"] is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: self.attach['decoder'] is not initialized."
            )

        bs = xs.size(0)
        logits, lx = self.encoder(xs, lx)

        # y_samples: (N, k, L), ly_samples: (N, k)
        y_samples, _, _, ly_samples = self.attach["decoder"].decode(
            logits.float().cpu(), lx.cpu()
        )

        """NOTE:
            for CTC training, we flatten the label seqs to 1-dim,
            so here we need to deal with that
        """
        if ys.dim() == 1:
            ground_truth = [t.cpu().tolist() for t in torch.split(ys, ly.tolist())]
        else:
            ground_truth = [ys[i, : ly[i]] for i in range(ys.size(0))]
        hypos = [y_samples[n, 0, : ly_samples[n, 0]].tolist() for n in range(bs)]

        return cal_wer(ground_truth, hypos)

    def wave2fbank(self,audio, lx):
        audio = audio[...,0]
        samples, flens = self.stft(audio, lx)
        input_power = samples[..., 0] ** 2 + samples[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        feats, _ = self.logmel(input_amp, flens)
        
        return feats,flens
    
    def beamforming(self,audio, lx):
        samples, flens = self.stft(audio, lx)
        samples, flens1 , _ =self.beamformer(samples,flens)
        assert (flens == flens1).all()
        
        # cal fbank
        input_power = samples[..., 0] ** 2 + samples[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        feats, _ = self.logmel(input_amp, flens)
        
        return feats, flens
    
    def forward(self, audio, lx, labels, ly):
        
        feats, flens = self.beamforming(audio, lx)
        
        
        if torch.isnan(feats).any():
            print("feats array contains NaN values!")

        
        logits, lx = self.encoder(feats, flens)
        logits = torch.log_softmax(logits, dim=-1)

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
        
        if float('inf') == loss:
            print("loss is inf!")
        
        if math.isnan(loss):
            print("loss is NaN!")    
        
        
        return loss


def cal_wer(gt: List[List[int]], hy: List[List[int]]) -> Tuple[int, int]:
    """compute error count for list of tokens"""
    assert len(gt) == len(hy)
    err = 0
    cnt = 0
    for i in range(len(gt)):
        err += Levenshtein.distance(
            "".join(chr(n) for n in hy[i]), "".join(chr(n) for n in gt[i])
        )
        cnt += len(gt[i])
    return (err, cnt)


@torch.no_grad()
def custom_evaluate(testloader, args: argparse.Namespace, manager: Manager) -> float:
    model = manager.model
    cnt_tokens = 0
    cnt_err = 0
    n_proc = dist.get_world_size()

    for minibatch in tqdm(
        testloader,
        desc=f"Epoch: {manager.epoch} | eval",
        unit="batch",
        disable=(args.gpu != 0),
        leave=False,
    ):
        feats, ilens, labels, olens = minibatch[:4]
        feats = feats.cuda(args.gpu, non_blocking=True)
        ilens = ilens.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)
        olens = olens.cuda(args.gpu, non_blocking=True)

        part_cnt_err, part_cnt_sum = model.module.get_wer(feats, labels, ilens, olens)
        cnt_err += part_cnt_err
        cnt_tokens += part_cnt_sum

    gather_obj = [None for _ in range(n_proc)]
    dist.gather_object(
        (cnt_err, cnt_tokens), gather_obj if args.rank == 0 else None, dst=0
    )
    dist.barrier()
    if args.rank == 0:
        l_err, l_sum = list(zip(*gather_obj))
        wer = sum(l_err) / sum(l_sum)
        manager.writer.add_scalar("loss/dev-token-error-rate", wer, manager.step)
        scatter_list = [wer]
    else:
        scatter_list = [None]

    dist.broadcast_object_list(scatter_list, src=0)
    return scatter_list[0]


def build_beamdecoder(cfg: dict) -> CTCBeamDecoder:
    """
    beam_size:
    num_classes:
    kenlm:
    alpha:
    beta:
    ...
    """

    assert "num_classes" in cfg, "number of vocab size is required."

    if "kenlm" in cfg:
        labels = [str(i) for i in range(cfg["num_classes"])]
        labels[0] = "<s>"
        labels[1] = "<unk>"
    else:
        labels = [""] * cfg["num_classes"]

    return CTCBeamDecoder(
        labels=labels,
        model_path=cfg.get("kenlm", None),
        beam_width=cfg.get("beam_size", 16),
        alpha=cfg.get("alpha", 1.0),
        beta=cfg.get("beta", 0.0),
        num_processes=cfg.get("num_processes", 6),
        log_probs_input=True,
        is_token_based=("kenlm" in cfg),
    )


def build_model(
    cfg: dict,
    args: Optional[Union[argparse.Namespace, dict]] = None,
    dist: bool = True,
    wrapper: bool = True,
) -> Union[nn.parallel.DistributedDataParallel, AMTrainer, model_zoo.AbsEncoder]:
    """
    for ctc-crf training, you need to add extra settings in
    cfg:
        trainer:
            use_crf: true/false,
            lamb: 0.01,
            den_lm: xxx

            decoder:
                beam_size:
                num_classes:
                kenlm:
                alpha:
                beta:
                ...
        ...
    """
    if "trainer" not in cfg:
        cfg["trainer"] = {}

    assert "encoder" in cfg
    netconfigs = cfg["encoder"]
    net_kwargs = netconfigs["kwargs"]  # type:dict

    # when immigrate configure from RNN-T to CTC,
    # one usually forget to set the `with_head=True` and 'num_classes'
    if "with_head" in net_kwargs and not net_kwargs["with_head"]:
        print(
            "WARNING: 'with_head' in field:encoder:kwargs is False, "
            "If you don't know what this means, set it to True."
        )

    if "num_classes" not in net_kwargs:
        raise Exception(
            "error: 'num_classes' in field:encoder:kwargs is not set. "
            "You should specify it according to your vocab size."
        )

    am_model = getattr(model_zoo, netconfigs["type"])(
        **net_kwargs
    )  # type: model_zoo.AbsEncoder
    if not wrapper:
        return am_model

    # initialize beam searcher
    if "decoder" in cfg["trainer"]:
        cfg["trainer"]["decoder"] = build_beamdecoder(cfg["trainer"]["decoder"])

    model = AMTrainer(am_model, **cfg["trainer"])
    if not dist:
        return model

    assert args is not None, f"You must tell the GPU id to build a DDP model."
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    elif not isinstance(args, dict):
        raise ValueError(f"unsupport type of args: {type(args)}")

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)

    model.cuda(args["gpu"])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args["gpu"]])
    return model


def _parser():
    parser = coreutils.basic_trainer_parser("CTC trainer.")
    parser.add_argument(
        "--eval-error-rate",
        action="store_true",
        help="Use token error rate for evaluation instead of CTC loss (default). "
        "If specified, you should setup 'decoder' in 'trainer' configuration.",
    )
    return parser


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)


if __name__ == "__main__":
    print(
        "NOTE:\n"
        "    since we import the build_model() function in cat.ctc,\n"
        "    we should avoid calling `python -m cat.ctc.train`, instead\n"
        "    running `python -m cat.ctc`"
    )
