# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Sardar (sar_dar@163.com)

"""Top interface of 3 network JSA training.
    S2P: CTC based speech to phone model.
    G2P: CTC based character to phone model, used to generate proposal
    P2G: CTC based phone to BPE model.
"""

__all__ = ["AMTrainer", "build_model", "_parser", "main"]

import re
import os
from typing import *

import argparse
import Levenshtein

import ctc_align
from ctcdecode import CTCBeamDecoder
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast
import torch.distributed as dist
import jiwer

from cat.shared.manager_jsa import Manager
from cat.shared import coreutils
from cat.shared import encoder as model_zoo
from cat.shared.data import JSASpeechDataset, JSAsortedPadCollateASR
from cat.shared.tokenizer import AbsTokenizer, load

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
        mkwargs["T_dataset"] = JSASpeechDataset

    if "collate_fn" not in mkwargs:
        mkwargs["collate_fn"] = JSAsortedPadCollateASR(flatten_target=False)

    if "func_build_model" not in mkwargs:
        mkwargs["func_build_model"] = build_model

    if "func_eval" not in mkwargs:
        mkwargs["func_eval"] = custom_evaluate

    mkwargs["args"] = args
    manager = Manager(**mkwargs)

    tr_dataset = manager.trainloader.dl.dataset
    coreutils.distprint(
        f"  total {tr_dataset.__len__()} utterances are used in training.", args.gpu
    )
    # training
    manager.run(args)

def subsetIndex(alist, blist):
    idx_list = []
    for idx, id in enumerate(alist):
        if id in blist:
            idx_list.append(idx)
    return idx_list

class AMTrainer(nn.Module):
    def __init__(
        self,
        s2p_encoder: model_zoo.AbsEncoder,
        phn_decoder: CTCBeamDecoder,
        p2g_encoder: model_zoo.AbsEncoder,
        p2g_decoder: model_zoo.AbsEncoder,
        g2p_encoder: model_zoo.AbsEncoder,
        bpe_tokenizer: AbsTokenizer = None,
        phone_tokenizer: AbsTokenizer = None,
        sampling: bool = True,
        n_samples: int = 10,
        cache_enabled: bool = False,
        supervised_trans: str = None,
    ):
        super().__init__()

        self.s2p_encoder = s2p_encoder
        self.phn_searcher = phn_decoder
        self.p2g_encoder = p2g_encoder
        self.p2g_decoder = p2g_decoder
        self.g2p_encoder = g2p_encoder
        self.n_samples = n_samples
        self.add_supervised = True if supervised_trans else False

        self.ctc_loss = nn.CTCLoss(reduction='none',zero_infinity=True)
        self.ctc_loss_for_eval = nn.CTCLoss()
        self.bpe_tokenizer = bpe_tokenizer

        if self.add_supervised:
            assert os.path.isfile(supervised_trans), "Spervised trans file is not found."
            self.supervised = {}
            with open(supervised_trans, 'r', encoding='utf-8') as f:
                for line in f:
                    uid, seq = re.split('\t| ', line.strip(), maxsplit=1)
                    self.supervised[uid] = torch.tensor(phone_tokenizer.encode(seq),dtype=torch.int32)

    def forward(self, x, lx, y, ly, y_char, ly_char, uids):
        # s2p_encoder forward
        logits_s2p_enc, logits_lens_s2p_enc = self.s2p_encoder(x, lx)
        logits_s2p_enc = torch.log_softmax(logits_s2p_enc, dim=-1)

        device = x.device
        batch_size = x.size(0)
        ly = ly.to(torch.int)
        acc_cnt = torch.zeros([1], dtype=torch.int32)
        acc_g2p_loss = torch.zeros([1], dtype=torch.float32, device=device)
        acc_s2p_loss = torch.zeros([1], dtype=torch.float32, device=device)
        acc_p2g_loss = torch.zeros([1], dtype=torch.float32, device=device)
        total_loss = torch.zeros([batch_size], dtype=torch.float64, device=device)
        acc_inf_lens = torch.zeros([1], dtype=torch.int32, device=device)

        # init z_old
        z_old = [torch.tensor([],dtype=torch.int32, device=device) for _ in range(batch_size)]
        zlens_old = torch.zeros([batch_size], dtype=torch.int32, device=device)
        p_old = torch.zeros([batch_size], dtype=torch.float32, device=device)

        # sampling
        logits_g2p_enc, logits_lens_g2p_enc = self.g2p_encoder(y_char, ly_char)
        logits_g2p_enc = torch.log_softmax(logits_g2p_enc, dim=-1)
        samples, sample_lens = self._sample(logits_g2p_enc.detach().exp(), logits_lens_g2p_enc)

        
        samples = samples.transpose(0,1).cuda()
        sample_lens = sample_lens.transpose(0,1).cuda()
        logits_s2p_enc = logits_s2p_enc.transpose(0, 1)
        logits_g2p_enc = logits_g2p_enc.transpose(0, 1)
        
        isFirstBeam = True
        for bantched_sample, zlens_new in zip(samples, sample_lens):
            z_new = [bantched_sample[batch, : zlens_new[batch]] for batch in range(batch_size)]
            z_new_in_batch, zlens_new_in_batch = self.validate_zlen_and_pad(z_new, zlens_new)
            with torch.no_grad():
                # g2p forward and calculate loss
                g2p_loss_new = self.ctc_loss(logits_g2p_enc, z_new_in_batch, logits_lens_g2p_enc.to('cpu', torch.int), zlens_new_in_batch.cpu()) / zlens_new_in_batch
                
                # s2p forward and calculate loss
                s2p_loss_new = self.ctc_loss(logits_s2p_enc, z_new_in_batch, logits_lens_s2p_enc.to('cpu', torch.int), zlens_new_in_batch.cpu()) / zlens_new_in_batch
                
                # p2g forward and calculate loss 
                logits_p2g_enc, logits_lens_p2g_enc = self.p2g_encoder(z_new_in_batch, zlens_new_in_batch)
                logits_p2g_enc = torch.log_softmax(logits_p2g_enc, dim=-1)
                p2g_loss_new = self.ctc_loss(logits_p2g_enc.transpose(0, 1), y, logits_lens_p2g_enc.to('cpu', torch.int), ly.cpu()) / ly
                inf_new = ~(g2p_loss_new == 0) * ~(s2p_loss_new == 0) * ~(p2g_loss_new == 0)

                # calculate importance weight
                p_new = g2p_loss_new - s2p_loss_new - p2g_loss_new
                
                # if z_old is empty, which means cache[uid] is empty
                if isFirstBeam and len(torch.nonzero(zlens_old == 0)) > (batch_size//4):
                    z_old = z_new       # List[tensor,tensor,...]
                    zlens_old = zlens_new
                    p_old = p_new  # tensor(int32) in cuda
                    isFirstBeam = False
                    acc_cnt += torch.tensor(batch_size, dtype=torch.int32)
                else:
                    # z_old from cache
                    if p_old is None:
                        z_old_in_batch, zlens_old_in_batch = self.validate_zlen_and_pad(z_old, zlens_old)
                        g2p_loss_old = self.ctc_loss(logits_g2p_enc, z_old_in_batch, logits_lens_g2p_enc.to('cpu', torch.int), zlens_old_in_batch.cpu()) / zlens_old_in_batch
                        s2p_loss_old = self.ctc_loss(logits_s2p_enc, z_old_in_batch, logits_lens_s2p_enc.to('cpu', torch.int), zlens_old_in_batch.cpu()) / zlens_old_in_batch
                        logits_p2g_enc_old, logits_lens_p2g_enc_old = self.p2g_encoder(z_old_in_batch.cuda(), zlens_old_in_batch.cuda())
                        logits_p2g_enc_old = torch.log_softmax(logits_p2g_enc_old, dim=-1)
                        p2g_loss_old = self.ctc_loss(logits_p2g_enc_old.transpose(0, 1), y, logits_lens_p2g_enc_old.to('cpu', torch.int), ly.cpu()) / ly
                        p_old = g2p_loss_old - s2p_loss_old - p2g_loss_old
                    accpet_index = self.accept_reject(p_old, p_new, zlens_old, ~inf_new)
                    if accpet_index.any():
                        for i in accpet_index:
                            z_old[i] = z_new[i]
                        zlens_old[accpet_index] = zlens_new[accpet_index]
                        p_old[accpet_index] = p_new[accpet_index]
                        acc_cnt += torch.tensor(accpet_index.shape[0], dtype=torch.int32)

            z_old_in_batch, zlens_old_in_batch = self.validate_zlen_and_pad(z_old, zlens_old.clone())
            logits_p2g, logits_lens_p2g = self.p2g_encoder(z_old_in_batch.cuda(), zlens_old_in_batch)
            logits_p2g = torch.log_softmax(logits_p2g, dim=-1).transpose(0, 1)
            p2g_loss = self.ctc_loss(logits_p2g, y, logits_lens_p2g.to('cpu', torch.int), ly.cpu()) / ly

            if self.add_supervised:
                z_old_for_grad, zlens_old_for_grad = self.replace_supervised(z_old, zlens_old, uids)
            else:
                z_old_for_grad, zlens_old_for_grad = z_old_in_batch, zlens_old_in_batch
            g2p_loss = self.ctc_loss(logits_g2p_enc, z_old_for_grad, logits_lens_g2p_enc.to('cpu', torch.int), zlens_old_for_grad.cpu()) / zlens_old_for_grad
            s2p_loss = self.ctc_loss(logits_s2p_enc, z_old_for_grad, logits_lens_s2p_enc.to('cpu', torch.int), zlens_old_for_grad.cpu()) / zlens_old_for_grad
            
            inf_old = ~(g2p_loss == 0) * ~(s2p_loss == 0) * ~(p2g_loss == 0)
    
            g2p_loss = g2p_loss * inf_old
            s2p_loss = s2p_loss * inf_old
            p2g_loss = p2g_loss * inf_old
            
            total_loss += s2p_loss
            total_loss += g2p_loss
            total_loss += p2g_loss
            acc_g2p_loss += torch.mean(g2p_loss.data[inf_old])
            acc_s2p_loss += torch.mean(s2p_loss.data[inf_old])
            acc_p2g_loss += torch.mean(p2g_loss.data[inf_old])
            acc_inf_lens += len(torch.nonzero(inf_old))
        
        return torch.sum(total_loss)/acc_inf_lens, torch.sum(acc_g2p_loss)/self.n_samples, torch.sum(acc_s2p_loss)/self.n_samples, torch.mean(acc_p2g_loss)/self.n_samples, acc_cnt / (self.n_samples * batch_size)
   
    def evaluate(self, x, lx, y, ly, y_pid, ly_pid):
        bs = x.size(0)
        logits_s2p_enc, logits_lens_s2p_enc = self.s2p_encoder(x, lx)
        logits_s2p_enc = torch.log_softmax(logits_s2p_enc, dim=-1)
        s2p_results, _, _, s2p_result_lens = self.phn_searcher.decode(logits_s2p_enc, logits_lens_s2p_enc)
        # best beam, beam_results: (bach, beam, T) -> (bach, T)
        s2p_results = s2p_results.transpose(0,1)[0]
        # s2p_result_lens: (bach, beam) -> (bach)
        s2p_result_lens = s2p_result_lens.transpose(0,1)[0]
        s2p_loss = self.ctc_loss_for_eval(logits_s2p_enc.transpose(0, 1), y_pid, logits_lens_s2p_enc.to('cpu', torch.int), ly_pid)
        
        # calculate p2g loss and p2g WER
        z_new = [s2p_results[i, : s2p_result_lens[i]] for i in range(bs)]
        z_new_in_batch = pad_sequence(z_new, batch_first=True, padding_value=0)
        logits_p2g, logits_lens_p2g = self.p2g_encoder(z_new_in_batch.cuda(), s2p_result_lens.cuda())
        logits_p2g = torch.log_softmax(logits_p2g, dim=-1)
        p2g_loss = self.ctc_loss_for_eval(logits_p2g.transpose(0, 1), y, logits_lens_p2g.to('cpu', torch.int), ly)
        p2g_results, _, _, p2g_result_lens = self.p2g_decoder.decode(logits_p2g, logits_lens_p2g)
        p2g_results = p2g_results.transpose(0,1)[0]
        p2g_result_lens = p2g_result_lens.transpose(0,1)[0]
        err_p2g, cnt_p2g = cal_wer([self.bpe_tokenizer.decode(y[i, : ly[i]].tolist()) for i in range(bs)],
                                   [self.bpe_tokenizer.decode(p2g_results[i, : p2g_result_lens[i]].tolist()) for i in range(bs)])
        
        ground_truth_phn = [y_pid[i, : ly_pid[i]].tolist() for i in range(bs)]
        err_s2p, cnt_s2p = cal_per(ground_truth_phn, [i.tolist() for i in z_new])
        
        ##########·DEBUG·CODE ###########
        # if torch.isinf(p2g_loss).any():
        #     p2g_loss_inf = self.ctc_loss(logits_p2g.transpose(0, 1), y, logits_lens_p2g.to('cpu', torch.int), ly) / ly.cuda()
        #     inf_index = torch.nonzero(p2g_loss_inf == 0)
        #     for i in inf_index:
        #         print(f"### Warning: p2g_loss got inf value in validation stpe, please check the data '{uids[inf_index]}'")
            # raise StopIteration
        ##########·DEBUG·CODE ###########        

        return s2p_loss, p2g_loss, err_s2p, cnt_s2p, err_p2g, cnt_p2g

    def _sample(self, probs, lx, n_samples=None):
        N, T, V = probs.shape
        K = n_samples if n_samples else self.n_samples
        # (NT, K)
        samples = torch.multinomial(probs.view(-1, V), K, replacement=True).view(
            N, T, K
        )
        # (N, T, K) -> (N, K, T) -> (N*K, T)
        ys, ly = ctc_align.align_(
            samples.transpose(1, 2).contiguous().view(-1, T),
            # (N, ) -> (N, 1) -> (N, K) -> (N*K, )
            lx.unsqueeze(1).repeat(1, K).contiguous().view(-1),
        )
        return ys.view(N, K, T), ly.view(N, K)

    def accept_reject(self, old, new, zlens_old, inf_new):
        e_new = torch.exp(new).cpu()
        e_old = torch.exp(old).cpu()
        MH_ratio = torch.div(e_new, e_old)
        rand = torch.rand(MH_ratio.size()[0])
        yes_no = MH_ratio > rand
        # Always accept when cache[uid] is empty
        yes_no[zlens_old == 0] = True
        # Always reject when loss is inf
        yes_no[inf_new] = False
        accpet_index = torch.nonzero(yes_no).squeeze(dim=1)
        return accpet_index
    
    def repeated(self, z_old, z_new, uids):
        for i in range(len(z_new)):
            if self.add_supervised and uids[i] in self.supervised.keys():
                continue
            if len(z_old[i]) != len(z_new[i]):
                return False
            dif_index = torch.nonzero(z_old[i] != z_new[i])
            if len(dif_index) > 0:
                return False
        return True

    def replace_supervised(self, z, zlens, uids):
        index = []
        device = zlens.device
        for idx, id in enumerate(uids):
            if id in self.supervised.keys():
                index.append(idx)
        if len(index) > 0:
            zlist_new = z.copy()
            zlens_new = zlens.clone()
            for i in index:
                zlist_new[i] = self.supervised[uids[i]].to(device)
                zlens_new[i] = torch.tensor([zlist_new[i].size(0)],dtype=torch.int32, device=device)
            return self.validate_zlen_and_pad(zlist_new, zlens_new)
        else:
            return self.validate_zlen_and_pad(z, zlens.clone())
    
    def validate_zlen_and_pad(self, zlist, zlens):
        if (zlens == 0).any():
            zlist_new = zlist.copy()
            zlens_new = zlens.clone()
            # (num_utt % batch_size) item not covered in cache because of mini-batch sampling
            index = torch.nonzero(zlens_new == 0).squeeze(dim=1)
            for i in index:
                zlens_new[i] = 1
                zlist_new[i] = torch.ones([1], dtype=torch.int32, device=zlens.device)

            return pad_sequence(zlist_new, batch_first=True, padding_value=0), zlens_new
        else:
            return pad_sequence(zlist, batch_first=True, padding_value=0), zlens
    
    def clean_unpickable_objs(self):
        pass

    def get_wer(
        self, type, xs: torch.Tensor, ys: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor
    ):
        acc_err = 0.
        acc_cnt = 0
        snt_er = []
        for x, xlen, y, ylen in zip(xs, lx, ys, ly):
            if type == "PER":
                x1 = [x[:xlen].tolist()]
                y1 = [y[:ylen].tolist()]
                err, cnt = cal_wer(y1, x1)
            elif type == "WER":
                x1 = [self.bpe_tokenizer.decode(x[:xlen].tolist())]
                y1 = [self.bpe_tokenizer.decode(y[:ylen].tolist())]
                measure = jiwer.compute_measures(y1, x1)
                cnt = measure['hits'] + measure['substitutions'] + measure['deletions']
                err = measure['substitutions'] + measure['deletions'] + measure['insertions']
            else:
                raise TypeError(f"type {type} is illegal!")
            acc_err += err
            acc_cnt += cnt
            snt_er.append(err / cnt)
        return torch.tensor([acc_err / acc_cnt], dtype=torch.float16), torch.tensor(snt_er, dtype=torch.float16)
    
def cal_wer(gt: List[List[str]], hy: List[List[str]], return_snt_wer: bool = False) -> Tuple[int, int]:
    """compute error count for list of tokens"""
    assert len(gt) == len(hy)
    acc_err = 0
    acc_cnt = 0
    snt_er = []
    for i in range(len(gt)):
        measure = jiwer.compute_measures(gt[i], hy[i])
        cnt = measure['hits'] + measure['substitutions'] + measure['deletions']
        err = measure['substitutions'] + measure['deletions'] + measure['insertions']
        acc_err += err
        acc_cnt += cnt
        snt_er.append(err / cnt)
        
    if return_snt_wer:
        return snt_er
    else:
        return acc_err, acc_cnt

def cal_per(gt: List[List[int]], hy: List[List[int]], return_snt_wer: bool = False) -> Tuple[int, int]:
    """compute error count for list of tokens"""
    assert len(gt) == len(hy)
    acc_err = 0
    acc_cnt = 0
    snt_er = []
    for i in range(len(gt)):
        err = Levenshtein.distance(
            "".join(chr(n) for n in hy[i]), "".join(chr(n) for n in gt[i])
        )
        cnt = len(gt[i])
        acc_err += err
        acc_cnt += cnt
        snt_er.append(err / cnt)

    if return_snt_wer:
        return snt_er
    else:
        return acc_err, acc_cnt

@torch.no_grad()
def custom_evaluate(testloader, args: argparse.Namespace, manager: Manager) -> float:
    model = manager.model
    cnt_err_s2p = 0
    cnt_tokens_s2p = 0
    cnt_err_p2g = 0
    cnt_tokens_p2g = 0
    total_s2p_loss = 0.0
    total_p2g_loss = 0.0
    cnt_seq = 0

    for minibatch in tqdm(
        testloader,
        desc=f"Epoch: {manager.epoch} | eval",
        unit="batch",
        disable=(args.gpu != 0),
        leave=False,
    ):
        feats, ilens, labels, olens, labels_pid, plens = minibatch[:6]
        feats = feats.cuda(args.gpu)
        ilens = ilens.cuda(args.gpu)
        
        with autocast(enabled=args.amp):
            s2p_loss, p2g_loss, err_s2p, cnt_s2p, err_p2g, cnt_p2g = model.module.evaluate(feats, ilens, labels, olens, labels_pid, plens)
        cnt_err_s2p += err_s2p
        cnt_tokens_s2p += cnt_s2p
        cnt_err_p2g += err_p2g
        cnt_tokens_p2g += cnt_p2g
        cnt_seq += feats.size(0)
        total_s2p_loss += s2p_loss.float() * feats.size(0)
        total_p2g_loss += p2g_loss.float() * feats.size(0)
    
    cnt_seq = total_s2p_loss.new_tensor(cnt_seq)
    cnt_err_s2p = total_s2p_loss.new_tensor(cnt_err_s2p)
    cnt_tokens_s2p = total_s2p_loss.new_tensor(cnt_tokens_s2p)
    cnt_err_p2g = total_s2p_loss.new_tensor(cnt_err_p2g)
    cnt_tokens_p2g = total_s2p_loss.new_tensor(cnt_tokens_p2g)
    dist.all_reduce(total_s2p_loss, dist.ReduceOp.SUM)
    dist.all_reduce(total_p2g_loss, dist.ReduceOp.SUM)
    dist.all_reduce(cnt_seq, dist.ReduceOp.SUM)
    dist.all_reduce(cnt_err_s2p, dist.ReduceOp.SUM)
    dist.all_reduce(cnt_tokens_s2p, dist.ReduceOp.SUM)
    dist.all_reduce(cnt_err_p2g, dist.ReduceOp.SUM)
    dist.all_reduce(cnt_tokens_p2g, dist.ReduceOp.SUM)
    avg_s2p_loss = (total_s2p_loss / cnt_seq).item()
    avg_p2g_loss = (total_p2g_loss / cnt_seq).item()
    s2p_per = (cnt_err_s2p / cnt_tokens_s2p).item()
    p2g_wer = (cnt_err_p2g / cnt_tokens_p2g).item()

    if args.rank == 0:
        manager.writer.add_scalar("dev_loss/S2P", avg_s2p_loss, manager.step)
        manager.writer.add_scalar("dev_loss/P2G", avg_p2g_loss, manager.step)
        manager.writer.add_scalar("dev_PER/S2P", s2p_per, manager.step)
        manager.writer.add_scalar("dev_WER/P2G", p2g_wer, manager.step)

    return p2g_wer

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

    assert "s2p_encoder" in cfg
    s2p_netconfigs = cfg["s2p_encoder"]
    s2p_net_kwargs = s2p_netconfigs["kwargs"]  # type:dict

    n_classes = s2p_net_kwargs.pop("n_classes")
    s2p_encoder = getattr(model_zoo, s2p_netconfigs["type"])(
        num_classes = n_classes, **s2p_net_kwargs
    )  # type: model_zoo.AbsEncoder

    # initialize beam searcher
    if "decoder" in cfg["trainer"]:
        cfg["trainer"]["decoder"] = build_beamdecoder(cfg["trainer"]["decoder"])
    
    assert "jsa" in cfg
    jsa_cfg = cfg["jsa"]

    assert "p2g_encoder" in cfg
    p2g_enc_configs = cfg["p2g_encoder"]
    p2g_enc_kwargs = p2g_enc_configs["kwargs"]  # type:dict
    p2g_encoder = getattr(model_zoo, p2g_enc_configs["type"])(**p2g_enc_kwargs)  # type: model_zoo.AbsEncoder

    assert "g2p_encoder" in cfg
    g2p_enc_configs = cfg["g2p_encoder"]
    g2p_enc_kwargs = g2p_enc_configs["kwargs"]  # type:dict
    n_classes = g2p_enc_kwargs.pop("n_classes")
    g2p_encoder = getattr(model_zoo, g2p_enc_configs["type"])(
        num_classes = n_classes, **g2p_enc_kwargs)  # type: model_zoo.AbsEncoder
    
    assert "beamDecoder" in cfg
    beamDecoder_cfg = cfg["beamDecoder"]
    phn_searcher = eval(beamDecoder_cfg["type"])(
        [""] * beamDecoder_cfg["n_classes"],
        beam_width=beamDecoder_cfg["beam_width"],
        log_probs_input=beamDecoder_cfg["log_probs_input"],
        num_processes=beamDecoder_cfg["num_processes"]
    )
    p2g_decoder = eval(beamDecoder_cfg["type"])(
            [""] * p2g_enc_kwargs["num_classes"],
            beam_width=beamDecoder_cfg["beam_width"],
            log_probs_input=beamDecoder_cfg["log_probs_input"],
            num_processes=beamDecoder_cfg["num_processes"]
        )
    
    if dist:
        # for training
        assert "tokenizer" in args, f"tokenizer is required for JSA training."
        bpe_tokenizer = load(args.tokenizer)
        assert "phone_tokenizer" in args, f"phone_tokenizer is required for JSA training."
        phone_tokenizer = load(args.phone_tokenizer)
    else:
        bpe_tokenizer = None
        phone_tokenizer = None
        jsa_cfg["supervised_trans"] = None

    if s2p_netconfigs.get("freeze", False):
        s2p_encoder.requires_grad_(False)
    if p2g_enc_configs.get("freeze", False):
        p2g_encoder.requires_grad_(False)
    if g2p_enc_configs.get("freeze", False):
        g2p_encoder.requires_grad_(False)

    model = AMTrainer(s2p_encoder,
                      phn_searcher,
                      p2g_encoder,
                      p2g_decoder,
                      g2p_encoder,
                      bpe_tokenizer,
                      phone_tokenizer,
                      **jsa_cfg)
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
    model = torch.nn.parallel.DistributedDataParallel(model, 
                                                      device_ids=[args["gpu"]],
                                                      find_unused_parameters=True)
                                                      
    init_checkpoint = OrderedDict()
    if "init_model" in s2p_netconfigs:
        coreutils.distprint(f"> initialize s2p_encoder from: {s2p_netconfigs['init_model']}", args["gpu"])
        s2p_enc_checkpoint = torch.load(
            s2p_netconfigs["init_model"], 
            map_location=f"cuda:{args['gpu']}"
        )["model"]  # type: OrderedDict
        s2p_enc_checkpoint = translate_checkpoint(s2p_enc_checkpoint, "encoder", "s2p_encoder")
        init_checkpoint.update(s2p_enc_checkpoint)
        del s2p_enc_checkpoint

    if "init_model" in p2g_enc_configs:
        coreutils.distprint(f"> initialize p2g_encoder from: {p2g_enc_configs['init_model']}", args["gpu"])
        p2g_enc_checkpoint = torch.load(
            p2g_enc_configs["init_model"], 
            map_location=f"cuda:{args['gpu']}"
        )["model"]  # type: OrderedDict
        p2g_enc_checkpoint = translate_checkpoint(p2g_enc_checkpoint, "encoder", "p2g_encoder")
        init_checkpoint.update(p2g_enc_checkpoint)
        del p2g_enc_checkpoint

    if "init_model" in g2p_enc_configs:
        coreutils.distprint(f"> initialize g2p_encoder from: {g2p_enc_configs['init_model']}", args["gpu"])
        g2p_enc_checkpoint = torch.load(
            g2p_enc_configs["init_model"], 
            map_location=f"cuda:{args['gpu']}"
        )["model"]  # type: OrderedDict
        g2p_enc_checkpoint = translate_checkpoint(g2p_enc_checkpoint, "encoder", "g2p_encoder")
        init_checkpoint.update(g2p_enc_checkpoint)
        del g2p_enc_checkpoint

    if len(init_checkpoint) != 0:
        model.load_state_dict(init_checkpoint, strict=False)
        del init_checkpoint

    return model

def translate_checkpoint(state_dict: OrderedDict, old_string: str, new_string: str) -> OrderedDict:
    """Translate checkpoint of previous version of RNN-T so that it could be loaded with the new one."""
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
