# Copyright 2025 Tsinghua University
# Apache 2.0.
# Author: Saier (sar_dar@163.com)

"""Top interface of CTC training.
"""

__all__ = ["AMTrainer", "build_model", "_parser", "main"]

from ..shared import Manager
from ..shared import coreutils
from ..shared import encoder as model_zoo
from ..shared.data import TonalKaldiSpeechDataset, sortedPadCollateTonalASR

import os
import argparse
import Levenshtein
from typing import *
from ctcdecode import CTCBeamDecoder
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast

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
        mkwargs["T_dataset"] = TonalKaldiSpeechDataset

    if "collate_fn" not in mkwargs:
        flatten_target = args.flatten_target_data if "flatten_target_data" in args else True
        mkwargs["collate_fn"] = sortedPadCollateTonalASR(flatten_target=flatten_target)

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
        # filter dev set 
        dev_dataset = manager.valloader.dataset
        orilen = len(dev_dataset)
        dev_dataset.filt_by_len(lambda x, y: x // SUBSAMPLING > y)
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
        decoder: CTCBeamDecoder = None,
        subClassList: List[str] = None,
        sub2base: Dict = None,
        loss_type: str = "latent_ctc",
        latent_map: Optional[Union[str, dict, Sequence[Sequence[int]]]] = None,
        major_label_offset: Optional[int] = None,
        blank_label: int = 0,
    ):
        super().__init__()

        self.encoder = encoder
        self.loss_type = loss_type
        self.is_latent_ctc = loss_type == "latent_ctc"
        self._latent_loss_kwargs = dict(
            latent_map=latent_map,
            major_label_offset=major_label_offset,
            blank_label=blank_label,
        )
        self.criterion = None
        if self.loss_type == "latent_ctc":
            # The criterion is resolved lazily. Decoding builds the same trainer
            # but never calls the loss, and the custom extension is not needed
            # there; training fails with an explicit message if it is missing.
            pass
        elif self.loss_type == "torch_ctc":
            self.criterion = nn.CTCLoss()
            print(
                "WARNING: 'torch_ctc' is not recommended for Latent Softmax training. Please use 'latent_ctc' instead."
            )
        else:
            raise ValueError(f"unsupported loss type: {self.loss_type}")
        
        self.attach = {"decoder": decoder}
        self.n_base_class = len(subClassList)
        self.subClassList = torch.tensor(subClassList, dtype=torch.long)
        self.sub_idx = self.subClassList.flatten()
        self.tiny_log_value = torch.tensor(-1e9, dtype=torch.float32)
        self.sub2base = sub2base

        if sub2base is None:
            raise ValueError(
                "AMTrainer: 'sub2base' is required for the latent-softmax model. "
                "It is normally injected by the CAT pipeline from the tonal tokenizer."
            )
        self.id_mapper = TensorIdMapper(sub2base)

    def _get_latent_criterion(self):
        """Resolve ``latent_ctc.LATENT_CTC_LOSS`` on first use.

        The extension is not needed for decoding, so the import is deferred and
        reported with an actionable message instead of failing at model
        construction time.
        """
        if self.criterion is None:
            try:
                from latent_ctc import LATENT_CTC_LOSS as LatentCTCLoss
            except (ImportError, ModuleNotFoundError) as e:
                raise ImportError(
                    "loss_type='latent_ctc' requires the 'latent_ctc' extension. "
                    "Build and install it from src/latent_ctc (`pip install .`), "
                    "or set trainer:loss_type to 'torch_ctc' to train Latent "
                    "Softmax with the pure-PyTorch aggregation path. "
                    "See src/latent_ctc/readme.md."
                ) from e
            self.criterion = LatentCTCLoss(**self._latent_loss_kwargs)
        return self.criterion

    def forward(self, feats, lx, labels, ly, non_tonal):
        logits, lx = self.encoder(feats, lx)
        if self.is_latent_ctc:
            criterion = self._get_latent_criterion()
            logits = torch.log_softmax(logits, dim=-1)
            labels = labels.cpu()
            lx = lx.cpu()
            ly = ly.cpu()
            with autocast(enabled=False):
                loss = criterion(
                    logits.float(),
                    labels.to(torch.int),
                    lx.to(torch.int),
                    ly.to(torch.int),
                )
        else:
            logits = self.latent_softmax(logits, non_tonal)
            logits = logits.transpose(0, 1)
            loss = self.criterion(
                logits, labels.to(torch.int), lx.to(torch.int), ly.to(torch.int)
            )
        return loss

    def latent_softmax(self, logits, non_tonal):
        logits = torch.log_softmax(logits, dim=-1) # [N, T, C]
        
        # create a new tensor with tiny value
        tiny_log_value = self.tiny_log_value.to(logits.dtype).to(logits.device)  # constant tensor
        extended_logits = logits.new_full(logits.shape[:-1] + (self.n_base_class,), tiny_log_value)
        if non_tonal.any():
            # For non-tonal classes, we need to sum the log prob of the sub-classes
            subClassList = self.subClassList.to(logits.device)
            extended_logits[non_tonal] = torch.cat(
                [
                    logits[non_tonal].index_select(dim=-1, index=sub).logsumexp(dim=-1, keepdim=True)
                    for sub in subClassList
                ],
                dim=-1,
            )
            
            # Set the logits corresponding to sub class to a very small value
            # logits_new[non_tonal].index_fill_(dim=-1,index=subClassList.flatten(),value=tiny_log_value)

            mask = torch.ones_like(logits, dtype=torch.bool)
            n_idx = torch.where(non_tonal)[0]
            mask[n_idx[:, None, None], torch.arange(mask.shape[1])[None, :, None], self.sub_idx[None, None, :]] = False
            logits_new = torch.where(mask, logits, tiny_log_value)
        else:
            logits_new = logits
        return torch.cat([logits_new, extended_logits], dim=-1)
    
    @torch.no_grad()
    def get_wer(
        self, xs: torch.Tensor, ys: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor, non_tonal: torch.Tensor
    ):
        if self.attach["decoder"] is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: self.attach['decoder'] is not initialized."
            )

        bs = xs.size(0)
        logits, lx = self.encoder(xs, lx)
        # logits = self.latent_softmax(logits, non_tonal)

        # y_samples: (N, k, L), ly_samples: (N, k)
        y_samples, _, _, ly_samples = self.attach["decoder"].decode(
            logits.float().cpu(), lx.cpu()
        )

        # Sardar
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
        # if non_tonal.any():
        #     non_tonal_index = torch.nonzero(non_tonal).flatten().tolist()
        #     for n in non_tonal_index:
        #         for s in range(len(hypos[n])):
        #             # if a phoneme is a sub-class, we need to replace to the base class
        #             if hypos[n][s] in self.sub2base:
        #                 hypos[n][s] = self.sub2base[hypos[n][s]]
        return cal_wer(ground_truth, hypos)
    
    def clean_unpickable_objs(self):
        # CTCBeamDecoder is unpickable,
        # So, this is required for inference.
        self.attach["decoder"] = None

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
        feats, ilens, labels, olens, non_tonal = minibatch
        feats = feats.cuda(args.gpu, non_blocking=True)
        ilens = ilens.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)
        olens = olens.cuda(args.gpu, non_blocking=True)
        non_tonal = non_tonal.cuda(args.gpu, non_blocking=True)

        part_cnt_err, part_cnt_sum = model.module.get_wer(feats, labels, ilens, olens, non_tonal)
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

    if args is None:
        raise RuntimeError(
            "build_model(cfg, args, dist=False) needs `args` so that "
            "'subClassList' and 'sub2base' can be resolved from --tokenizer. "
            "Use cat.ctc.decode_latent_softmax as the inference entry, which "
            "forwards its namespace."
        )

    # initialize beam searcher
    if "decoder" in cfg["trainer"] and dist:
        # if "test_vocab_size" not in args:
        #     raise Exception("error: 'test_vocab_size' is not set. ")
        # cfg["trainer"]["decoder"]["num_classes"] = args.test_vocab_size
        cfg["trainer"]["decoder"] = build_beamdecoder(cfg["trainer"]["decoder"])
    
    # During training the CAT pipeline injects both fields from the tonal
    # tokenizer. Decoding is launched from the CLI, where only --tokenizer is
    # available, so resolve the subclass structure from it as well.
    if ("subClassList" not in args or "sub2base" not in args) and getattr(
        args, "tokenizer", None
    ):
        from ..shared import tokenizer as tknz

        _tknz = tknz.load(args.tokenizer)
        args.subClassList = _tknz.get_subClassList
        args.sub2base = _tknz.get_sub2base

    if "subClassList" not in args:
        raise Exception("error: 'subClassList' is not set. ")
    if "sub2base" not in args:
        raise Exception("error: 'sub2base' is not set. ")

    model = AMTrainer(am_model, **cfg["trainer"],subClassList=args.subClassList, sub2base=args.sub2base)
    # model = AMTrainer(am_model, **cfg["trainer"])  
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
