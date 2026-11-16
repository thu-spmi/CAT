# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""
Transducer trainer.
"""

__all__ = ["TransducerTrainer", "build_model", "_parser", "main"]


from ..shared import Manager
from ..shared import coreutils
from ..shared import encoder as model_zoo
from ..shared.data import QwenP2GDataset, sortedPadCollateQwenP2G

import argparse
import Levenshtein
from collections import OrderedDict
from typing import *
from tqdm import tqdm



import torch
import torch.nn as nn
import torch.distributed as dist


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
        mkwargs["T_dataset"] = QwenP2GDataset

    from ..shared.tokenizer import load
    tokenizer = load(args.tokenizer)
    if "collate_fn" not in mkwargs:
        mkwargs["collate_fn"] = sortedPadCollateQwenP2G(pad_token_id=tokenizer.eos_token_id)
       
    if "func_build_model" not in mkwargs:
        mkwargs["func_build_model"] = build_model

    if "_wds_hook" not in mkwargs:
        mkwargs["_wds_hook"] = filter_hook

    # if (
    #     "func_eval" not in mkwargs
    #     and hasattr(args, "eval_error_rate")
    #     and args.eval_error_rate
    # ):
    #     mkwargs["func_eval"] = custom_evaluate

    mkwargs["args"] = args
    manager = Manager(**mkwargs)

    # set number of input samples used in an epoch
    # tr_dataset = manager.trainloader.dl.dataset
    manager.trainloader.dl.dataset.set_n_samples(args.n_samples)
    manager.trainloader.dl.dataset.set_tokenizer_and_prompt(tokenizer=tokenizer, prompt=args.prompt)
    manager.valloader.dataset.set_tokenizer_and_prompt(tokenizer=tokenizer, prompt=args.prompt)
    # training
    manager.run(args)


class QwenTrainer(nn.Module):
    def __init__(self,
                 model: model_zoo.Qwen3FromPretrainedModel
                 ):
        super().__init__()

        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    
    def forward(self, indexes, x, attn_mask, shift_labels):

        # indexes is a placeholder for Manager calculate global batch size
        B_K, T = x.shape                                     # batch * num_samples, seq_len
        B = indexes.size(0)                                  # batch size
        K = B_K // B                                         # num_samples    

        # print(f'B: {B},T: {T},K: {K}')

        logits = self.model(x, None, attn_mask)
        loss = self.criterion(logits.view(B_K*T, -1), shift_labels.view(-1))    # shape (B*K*T)
        loss = torch.sum(loss.view(B, K, T), dim=-1)        # get seq_loss, shape (B, K)
        return torch.mean(-torch.logsumexp(-loss, dim=-1))
    

    @torch.no_grad()
    def get_wer(
        self, xs: torch.Tensor, lx: None, ys: torch.Tensor, attn_mask: torch.Tensor
    ):
        # if self.attach["decoder"] is None:
        #     raise RuntimeError(
        #         f"{self.__class__.__name__}: self.attach['decoder'] is not initialized."
        #     )

        model_out = self.model(xs, None, attn_mask)
        logits = torch.log_softmax(model_out, dim = -1, dtype=torch.float32)
        hypos = torch.argmax(logits, dim = -1).tolist()
        gt = ys.tolist()
        # print(f'ground_turth{gt}')
        # print(f'hypos{hypos}')

        return cal_wer(gt, hypos)


    def generate_attention_mask(self, lens_matrix):
        seq_len = torch.max(lens_matrix) 
        attention_mask = torch.arange(seq_len, device=lens_matrix.device)[
            None, :
        ] < lens_matrix[:, None].to(lens_matrix.device)
        # attention_mask = (1.0 - attention_mask.to(self.dtype)) * torch.finfo(self.dtype).min
        return attention_mask.to(self.dtype)

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
        feats, ilens, labels, attn_mask = minibatch[:4]
        feats = feats.cuda(args.gpu, non_blocking=True)
        ilens = ilens.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)
        attn_mask = attn_mask.cuda(args.gpu, non_blocking=True)

        part_cnt_err, part_cnt_sum = model.module.get_wer(feats, None, labels, attn_mask)
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


def build_model(
    cfg: dict,
    args: Optional[Union[argparse.Namespace, dict]] = None,
    dist: bool = True,
    wrapper: bool = True,
) -> Union[nn.parallel.DistributedDataParallel, QwenTrainer]:
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
    
    assert "model" in cfg["trainer"]
    netconfigs = cfg["trainer"]["model"]
    net_kwargs = netconfigs["kwargs"]
    qwen_model = getattr(model_zoo, netconfigs["type"])(
        **net_kwargs
    ) 

    # trainer_cfg = cfg["trainer"]["kwargs"]

    if args.gradient_checkpointing:
        from cat.shared.lora import make_model_gradient_checkpointing_compatible
        qwen_model.model = make_model_gradient_checkpointing_compatible(qwen_model.model)
        qwen_model.model.gradient_checkpointing_enable()
        qwen_model.model.config.use_cache = False
    
    model = QwenTrainer(qwen_model)

    model = model.to(torch.bfloat16)

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
    parser = coreutils.basic_trainer_parser("Seq2Seq trainer.")
    parser.add_argument(
        "--eval-error-rate",
        action="store_true",
        help="Use token error rate for evaluation instead of AED loss (default). "
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
        "    since we import the build_model() function in cat.s2p2g,\n"
        "    we should avoid calling `python -m cat.s2p2g.train`, instead\n"
        "    running `python -m cat.s2p2g`"
    )
