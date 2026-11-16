# Copyright 2026 Tsinghua University
# Apache 2.0.
# Author: Saier (sar_dar@foxmail.com)

"""Top interface of CTC training (code-switching baseline, non-tonal dev PER).

This file is a copy of :mod:`cat.ctc.train`. The CTC objective, the encoder, the
data pipeline and the optimiser/scheduler handling are identical; the only
difference is the dev-set metric.

When ``--eval-error-rate`` is enabled, the decoded hypotheses **and** the
reference phoneme ids are both mapped from tone-marked subclass ids onto their
base vowels before scoring. The dev number is therefore a **non-tonal PER**
(NT-PER), which makes the dev PER curve directly comparable with the Latent
Softmax experiments: both monitor progress in the shared base-vowel space,
whereas a plain CTC baseline would report a tone-marked PER.

If you want the ordinary (tone-marked) dev PER instead, use
``cat/ctc/train.py``: set ``train:bin`` in ``hyper-p.json`` back to
``cat.ctc.train`` (and ``inference.infer.option.built-model-by`` too, where it
points at this module).

The subclass -> base-vowel mapping is read from the file named by
``train:option:sub2base`` in ``hyper-p.json``. It is a data asset produced per
tokenizer, not part of the repository; see the experiment ``readme.md`` for its
format.
"""

__all__ = ["AMTrainer", "build_model", "_parser", "main"]

from ..shared import Manager
from ..shared import coreutils
from ..shared import encoder as model_zoo
from ..shared.data import KaldiSpeechDataset, sortedPadCollateASR

import os
import json
import argparse
import Levenshtein
from typing import *
from ctcdecode import CTCBeamDecoder
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast
from dataclasses import dataclass

# NOTE (huahuan):
#   1/4 subsampling is used for Conformer model defaultly
#   for other sampling ratios, you need to modify the value.
#   Commonly, you can use a relatively larger value for allowing some margin.
SUBSAMPLING = 4

def normalize_mapping(mapping: Mapping[Union[int, str], Union[int, str]]) -> dict[int, int]:
    normalized = {}
    for key, value in mapping.items():
        normalized[int(key)] = int(value)
    return normalized

@dataclass(frozen=True)
class TensorIdMapper:
    mapping: Mapping[Union[int, str], Union[int, str]]
    def __post_init__(self) -> None:
        object.__setattr__(self, "mapping", normalize_mapping(self.mapping))

    def build_lut(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        max_id = max(max(self.mapping), max(self.mapping.values()))
        lut = torch.arange(max_id + 1, device=device, dtype=dtype)
        keys = torch.tensor(list(self.mapping.keys()), device=device, dtype=torch.long)
        values = torch.tensor(list(self.mapping.values()), device=device, dtype=dtype)
        lut[keys] = values
        return lut

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(ids) and not torch.is_complex(ids):
            output_dtype = ids.dtype
        else:
            raise TypeError("ids must be an integer tensor")

        lut = self.build_lut(ids.device, output_dtype)
        output = ids.clone()

        valid_mask = (ids >= 0) & (ids < lut.numel())
        output[valid_mask] = lut[ids[valid_mask].long()]
        return output
    
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
    if "T_dataset" in args:
        exec("from ..shared.data import " + args.T_dataset)
        mkwargs["T_dataset"] = eval(args.T_dataset)

    if "T_dataset" not in mkwargs:
        mkwargs["T_dataset"] = KaldiSpeechDataset

    if "collate_fn" not in mkwargs:
        flatten_target = args.flatten_target_data if "flatten_target_data" in args else True
        mkwargs["collate_fn"] = sortedPadCollateASR(flatten_target=flatten_target)

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
        # filter dev set 
        dev_dataset = manager.valloader.dataset
        orilen = len(dev_dataset)
        dev_dataset.filt_by_len(lambda x, y: x // SUBSAMPLING > y)
        if len(dev_dataset) < orilen:
            coreutils.distprint(
                f"  filtered {orilen-len(dev_dataset)} utterances in dev set.", args.gpu
            )
        coreutils.distprint(
            f"  total {dev_dataset.__len__()} utterances are used in validating.", args.gpu
        )
        # filter train set
        tr_dataset = manager.trainloader.dl.dataset
        orilen = len(tr_dataset)
        tr_dataset.filt_by_len(lambda x, y: x // SUBSAMPLING > y)
        if len(tr_dataset) < orilen:
            coreutils.distprint(
                f"  filtered {orilen-len(tr_dataset)} utterances in train set.", args.gpu
            )
        coreutils.distprint(
            f"  total {tr_dataset.__len__()} utterances are used in training.", args.gpu
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
        sub2base: Optional[Union[str, Dict]] = None,
    ):
        """
        Args:
            sub2base (str | dict, optional): subclass -> base-vowel id mapping
                used for the non-tonal dev PER. Either a path to the JSON file
                given by ``train:option:sub2base`` in ``hyper-p.json``, or an
                already-loaded dict. When omitted the model still trains and
                decodes; only the dev metric becomes unavailable.
        """
        super().__init__()

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

        # The mapping is a property of the experiment, not of the code, so it
        # comes from `train:option:sub2base` in hyper-p.json instead of a
        # hard-coded dataset path.
        if isinstance(sub2base, str):
            if not os.path.isfile(sub2base):
                raise FileNotFoundError(
                    f"AMTrainer: sub2base='{sub2base}' is not a valid file. "
                    "Set 'train:option:sub2base' in hyper-p.json."
                )
            with open(sub2base, "r", encoding="utf-8") as f:
                sub2base = json.load(f)
        self.sub2base = sub2base
        self.id_mapper = None if sub2base is None else TensorIdMapper(sub2base)

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

        if self.id_mapper is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: the non-tonal dev PER needs the "
                "subclass -> base mapping. Set 'train:option:sub2base' in "
                "hyper-p.json, or train with cat.ctc.train to report the "
                "tone-marked dev PER instead."
            )
        # Both sides are mapped onto base vowels, so a tone-marked hypothesis is
        # scored against a tone-marked reference in the shared base space.
        y_samples = self.id_mapper(y_samples)
        ys = self.id_mapper(ys)

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

    def forward(self, feats, lx, labels, ly):
        logits, lx = self.encoder(feats, lx)
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
        log_probs_input=True
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

    # The mapping is injected from hyper-p.json (`train:option:sub2base`) by the
    # CAT pipeline. `cat.ctc.decode` calls build_model() without args; the
    # trainer is then built without the mapping, which is fine because decoding
    # never computes the dev metric.
    if isinstance(args, argparse.Namespace):
        sub2base = getattr(args, "sub2base", None)
    elif isinstance(args, dict):
        sub2base = args.get("sub2base", None)
    else:
        sub2base = None

    # initialize beam searcher
    if "decoder" in cfg["trainer"]:
        cfg["trainer"]["decoder"] = build_beamdecoder(cfg["trainer"]["decoder"])

    model = AMTrainer(am_model, **cfg["trainer"], sub2base=sub2base)
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

    if "init_model_P2G" in args:
        assert os.path.isfile(args["init_model_P2G"])
        coreutils.distprint("> initialize p2g_encoder from: " + args["init_model_P2G"], args["gpu"])
        enc_checkpoint = torch.load(
            args["init_model_P2G"], 
            map_location=f"cuda:{args['gpu']}"
        )["model"]  # type: OrderedDict
        enc_checkpoint = translate_checkpoint(enc_checkpoint, "p2g_encoder", "encoder")
        model.load_state_dict(enc_checkpoint)
        del enc_checkpoint

    return model

def translate_checkpoint(state_dict: OrderedDict, old_string: str, new_string: str) -> OrderedDict:
    """Translate checkpoint of previous version of model so that it could be loaded with the new one."""
    old_string = old_string + '.'
    new_string = new_string + '.'
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if old_string in k:
            k = k.replace(old_string, new_string, 1)
            new_state_dict[k] = v
    return new_state_dict

def _parser():
    parser = coreutils.basic_trainer_parser("CTC trainer.")
    parser.add_argument(
        "--eval-error-rate",
        action="store_true",
        help="Use token error rate for evaluation instead of CTC loss (default). "
        "If specified, you should setup 'decoder' in 'trainer' configuration.",
    )
    parser.add_argument(
        "--sub2base",
        type=str,
        default=None,
        help="JSON file mapping tone-marked subclass ids to their base-vowel ids. "
        "Only used for the non-tonal dev PER; normally set through "
        "'train:option:sub2base' in hyper-p.json.",
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
