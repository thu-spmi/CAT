# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Hong Liu (liuhong21@mails.tsinghua.edu.cn)

from ...shared.decoder import *
from ...shared.encoder import *
from ...shared import tokenizer as tknz

import pickle
import math
import numpy as np
from typing import *
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from cat.shared import coreutils
from cat.lm import lm_builder


class EBM(nn.Module):
    def __init__(
        self,
        noise_rate: float = 1.0,  # rate of noise data number/ real data number
        method: Literal["nce", "dnce"] = "nce",  # nce or dynamic nce
        energy_func: str = "sumtargetlogit",  # 'hidden2scalar'/'logsumexplogit'/'maxlogit'/'sumtargetlogit'
        episilon: float = 1e-30,  # min of log()
        config_noise_model: str = None,  # noise configuration file path
        config_ebm_model: str = None,  # TRF model configuration file path
        check_ebm_model: str = None,  # load energy model from this checkpoint if its not None
        check_noise_model: str = None,  # load noise model from this checkpoint if its not None
        noise_score: bool = False,
        zeta_factor: float = 0,
        greedy_sampling: bool = False,
        tokenizer_path: str = None,
        bert_tokenizer: bool = False,  # the data is encoded by bert tokenizer, with [CLS] in the head and [SEP] in the tail
    ):
        super().__init__()
        # assign settings for EBM training
        self.noise_rate = noise_rate
        self.energy_func = energy_func
        self.episilon = episilon
        self.tokenizer = tknz.load(tokenizer_path) if tokenizer_path else None
        self.bert_tokenizer = bert_tokenizer
        self.method = method
        self.zeta_factor = zeta_factor
        self.greedy_sampling = greedy_sampling
        self.noise_score = noise_score

        # initialize trf and noise model
        noise_config = coreutils.readjson(config_noise_model)
        self.noise_type = list(noise_config.keys())[0]
        self.noise_cls = noise_config["decoder"]["type"]
        self.ebm_config = coreutils.readjson(config_ebm_model)
        self.nn_type = list(self.ebm_config.keys())[0]  # encoder or decoder
        if check_ebm_model is not None:
            trf_model = lm_builder(self.ebm_config, dist=False)
            coreutils.load_checkpoint(trf_model, check_ebm_model)
            self.udlying_nn = trf_model.lm
        else:
            model_cls = eval(self.ebm_config[self.nn_type]["type"])
            self.udlying_nn = model_cls(**self.ebm_config[self.nn_type]["kwargs"])

        if check_noise_model is not None:
            nlm = lm_builder(noise_config, dist=False)
            coreutils.load_checkpoint(nlm, check_noise_model)
            self.noise_model = nlm.lm
        else:
            model_cls = eval(noise_config["decoder"]["type"])
            self.noise_model = model_cls(**noise_config["decoder"]["kwargs"])
        if method == "nce":
            # freeze noise model if nce training
            self.noise_model.requires_grad_(False)
            self.noise_module = [self.noise_model]
            self.noise_model = None
        else:
            self.noise_module = [self.noise_model]

        if "hidden2scalar" in self.energy_func:
            hidden_size = (
                self.udlying_nn.config.hidden_size
                if hasattr(self.udlying_nn, "config")
                else self.udlying_nn.dim_hid
            )
            self.energy_lin = nn.Linear(in_features=hidden_size, out_features=1)
            if hasattr(self.udlying_nn, "model") and hasattr(
                self.udlying_nn.model, "pooler"
            ):
                self.udlying_nn.model.pooler = None

    # get NN feature
    def get_logit_feat(self, input_ids, logits, targets, in_lens: torch.LongTensor):
        # targets: (N, L, K)
        # logits: the output of nnlm, (N, L, V)
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)
        w = logits.gather(index=targets, dim=-1)
        # find the length and mask the tail
        padding_mask = torch.arange(input_ids.size(1), device=input_ids.device)[
            None, :
        ] < in_lens[:, None].to(input_ids.device)
        padding_mask = padding_mask.unsqueeze(2)
        w *= padding_mask
        # w: NN feature of the N sentences in this batch
        # w: (N, L, K)
        return w

    def noisem_score(self, seqs, in_lens, targets):
        if targets.dim() == 3:
            targets = targets.squeeze(2)
        noise_device = next(self.noise_module[0].parameters()).device
        log_probs = self.noise_module[0].score(
            seqs.to(noise_device), targets.to(noise_device), in_lens.to(noise_device)
        )
        return log_probs

    def score(
        self,
        input_ids: torch.LongTensor,
        targets: torch.LongTensor,
        in_lens: torch.Tensor,
        *args
    ):
        if self.bert_tokenizer and input_ids[0][0] == 0:
            input_ids = input_ids[:, 1:]  # delete 0 in the head
            targets = targets[:, 1:]
            in_lens -= 1
        if self.noise_score:
            score = self.noisem_score(input_ids, in_lens, targets)
        else:
            energy = self.calculate_energy(input_ids, targets, in_lens)
            score = -energy
        return score

    def getnoise(self, noise_num, maxlennoise=40):
        with torch.no_grad():
            noise = torch.zeros(
                [noise_num, maxlennoise],
                device=next(self.noise_module[0].parameters()).device,
                dtype=torch.long,
            )
            ones = torch.ones(
                [noise_num],
                device=next(self.noise_module[0].parameters()).device,
                dtype=torch.long,
            )
            if self.bert_tokenizer:
                # initialize the start token id with [CLS] id (101)
                noise_next = 101 * torch.ones(
                    [noise_num, 1],
                    device=next(self.noise_module[0].parameters()).device,
                    dtype=torch.long,
                )
                noise[:, 0] = 101 * torch.ones(
                    [noise_num],
                    device=next(self.noise_module[0].parameters()).device,
                    dtype=torch.long,
                )
            else:
                noise_next = torch.zeros(
                    [noise_num, 1],
                    device=next(self.noise_module[0].parameters()).device,
                    dtype=torch.long,
                )
            cache = None
            is_end = torch.zeros([noise_num], dtype=torch.bool, device=noise.device)
            lennoise = torch.ones([noise_num], dtype=torch.long, device=noise.device)
            end_mark = 102 if self.bert_tokenizer else 0
            for i in range(maxlennoise - 1):
                if self.noise_cls == "PretrainedTransformer":
                    noise_out, cache = self.noise_module[0](
                        noise_next, cache=cache, use_cache=True
                    )
                else:
                    noise_out, cache = self.noise_module[0](
                        src_ids=noise_next, cache=cache, input_lengths=ones
                    )
                noise_out = noise_out[:, -1, :]  # (B,V)
                noise_distribution = F.softmax(noise_out, dim=-1)
                if self.greedy_sampling:
                    noise_next = noise_distribution.argmax(-1).unsqueeze(-1)
                else:
                    noise_next = torch.multinomial(
                        noise_distribution, 1, True
                    )  # (B, 1)
                noise[:, i + 1] = noise_next.squeeze(1)
                lennoise += ones * (~is_end)
                is_end |= noise_next.squeeze(-1) == end_mark
                if all(is_end):
                    break

            padding_mask = torch.arange(noise.size(1), device=noise.device)[
                None, :
            ] < lennoise[:, None].to(noise.device)
            noise *= padding_mask
            targets = torch.cat(
                (
                    noise[:, 1:],
                    torch.zeros([noise_num, 1], device=noise.device, dtype=torch.long),
                ),
                dim=1,
            )

        return noise, lennoise, targets

    def cal_loss(self, inputs: torch.Tensor, energy_values, in_lens, targets):
        data_sample_num = energy_values.shape[0]
        noise_sample_num = int(data_sample_num * self.noise_rate)

        if targets.dim() == 2:
            targets = targets.unsqueeze(2)

        log_pm = -energy_values + self.zeta_factor * in_lens
        ppl_data = torch.exp(-log_pm.sum() / in_lens.sum())
        log_pn = self.noisem_score(inputs, in_lens, targets)
        with torch.no_grad():
            p1 = torch.sigmoid(math.log(self.noise_rate) - log_pm + log_pn)
        loss_data = -(p1 * log_pm).mean(dim=0)
        loss_data_true = -torch.mean(torch.log(1 - p1 + self.episilon))

        seqs, seqlens, seqtars = self.getnoise(noise_sample_num)
        log_pm_noise = -self.calculate_energy(seqs, seqtars, seqlens)
        with torch.no_grad():
            log_pn_noise = self.noisem_score(seqs, seqlens, seqtars)
            p0 = torch.sigmoid(-math.log(self.noise_rate) + log_pm_noise - log_pn_noise)
        loss_noise = self.noise_rate * (p0 * log_pm_noise).mean(dim=0)
        loss_noise_true = -self.noise_rate * torch.mean(
            torch.log(1 - p0 + self.episilon)
        )
        acc_data = (p1.data < 0.5).sum() / data_sample_num
        acc_noise = (p0.data < 0.5).sum() / noise_sample_num
        loss_noisem_ml = -log_pn.sum() / in_lens.sum() if self.method == "dnce" else 0
        loss = loss_data + loss_noise + loss_noisem_ml
        return loss, {
            "train/loss_data": loss_data.detach(),
            "train/loss_noise": loss_noise.detach(),
            "train/acc_data": acc_data.detach(),
            "train/acc_noise": acc_noise.detach(),
            "train/loss_true": (loss_data_true + loss_noise_true).detach(),
            "train/ppl_data": ppl_data.detach(),
        }

    def calculate_energy(self, inputs, targets, input_lengths: torch.LongTensor):
        if self.energy_func == "sumtargetlogit":
            # only this type will calculate energy per token
            # so we can use token-level discrete feature
            if targets.dim() == 2:
                targets = targets.unsqueeze(2)

            nn_logits, _ = self.udlying_nn(inputs, input_lengths=input_lengths)
            features = self.get_logit_feat(inputs, nn_logits, targets, input_lengths)
            padding_mask = (
                torch.arange(inputs.size(1), device=inputs.device)[None, :]
                < input_lengths[:, None].to(inputs.device)
            ).unsqueeze(2)
            energy = -(features * padding_mask).sum(dim=1).squeeze(1)
        # Note: the nn model must be BERT for the following 3 energy functions
        elif "hidden2scalar" in self.energy_func:
            # elif self.energy_func=='hidden2scalar':
            # TODO: add input length
            # if self.energy_func=='hidden2scalar-sum':
            hiddens, _ = self.udlying_nn(inputs, input_lengths=input_lengths)
            padding_mask = torch.arange(inputs.size(1), device=inputs.device)[
                None, :
            ] < input_lengths[:, None].to(inputs.device)
            energy = self.energy_lin(hiddens).squeeze(-1)  # (B, T)
            energy = (energy * padding_mask).sum(-1)  # (B, )

            # else: # default: use the hidden state of [CLS] to represent the sentence hidden
            #     pass
            # outputs = self.udlying_nn(inputs, input_lengths=input_lengths)
            # assert 'pooler_output' in outputs, 'The outputs has no attribute pooler_output'
            # hidden = outputs.pooler_output # (B,H)
            # energy = self.energy_lin(hidden).squeeze(-1) # (B,)

        elif self.energy_func == "logsumexplogit":
            logits, _ = self.udlying_nn(inputs, input_lengths=input_lengths)
            logit = logits[:, 0, :]  # (B, Classes)
            energy = -torch.logsumexp(logit, dim=-1)
        elif self.energy_func == "maxlogit":
            logits, _ = self.udlying_nn(inputs, input_lengths=input_lengths)
            logit = logits[:, 0, :]  # (B, Classes)
            energy = -torch.max(logit, dim=-1)
        elif self.energy_func == "summasklogit":
            # mask each token and obtain its logit on the original token
            # then sum all mask logits. (only for bert with LM head)
            # This energy function is more time-consuming than others
            energy = inputs.new_zeros(inputs.shape)
            for t in range(1, inputs.shape[1]):
                masked_inputs = inputs.clone()
                masked_inputs[:, t] = 103 * torch.ones(
                    [inputs.shape[0]], device=inputs.device, dtype=torch.long
                )
                logits, _ = self.udlying_nn(masked_inputs, input_lengths=input_lengths)
                logit = logits[:, t, :]  # (B, V)
                energy[:, t] = -logit.gather(
                    index=inputs[:, t].unsqueeze(1), dim=-1
                ).squeeze()  # (B,)
                # energy += -logit.gather(index=inputs[:, t].unsqueeze(1), dim=-1).squeeze() #(B,)
            padding_mask = torch.arange(inputs.size(1), device=inputs.device)[
                None, :
            ] < input_lengths[:, None].to(inputs.device)
            energy = (energy * padding_mask).sum(-1)  # (B,)
        elif self.energy_func == "sumtokenlogit":
            logits, _ = self.udlying_nn(inputs, input_lengths=input_lengths)
            energy = -logits.gather(
                index=inputs.unsqueeze(-1), dim=-1
            ).squeeze()  # (B, T)
            padding_mask = torch.arange(inputs.size(1), device=inputs.device)[
                None, :
            ] < input_lengths[:, None].to(inputs.device)
            energy = (energy * padding_mask).sum(-1)  # (B, )
        else:
            raise RuntimeError
        return energy  # shape: (B,)

    def forward(self, inputs, targets, input_lengths: torch.LongTensor):
        return self.calculate_energy(inputs, targets, input_lengths)


class TRFLM(EBM):
    def __init__(
        self,
        f_linfo: str = None,  # length information file
        alpha: float = 0.25,  # Interpolated coefficients alpha in dnce
        with_end_mark: bool = True,  # whether each sentence has an end mark
        **kwargs
    ):
        super().__init__(**kwargs)

        # assign settings for TRF nce training
        self.alpha = alpha
        self.with_end_mark = with_end_mark

        # initialize Pi and Zeta for TRF model
        with open(f_linfo, "rb") as fib:
            linfo = pickle.load(fib)
        self.max_len = linfo["max_len"]
        self.num_classes = (
            self.udlying_nn.config.vocab_size
            if self.ebm_config[self.nn_type]["type"] == "PretrainedTransformer"
            else self.ebm_config[self.nn_type]["kwargs"]["num_classes"]
        )
        if self.zeta_factor is None:
            # no zeta factor specified, use log(vocab_size) as zeta factor
            self.zeta = nn.Parameter(
                np.log(self.num_classes)
                * torch.tensor(range(-1, self.max_len - 1), dtype=torch.float32)
            )
        else:
            self.zeta = nn.Parameter(
                self.zeta_factor
                * torch.tensor(range(-1, self.max_len - 1), dtype=torch.float32)
            )
        self.zeta[0].data.zero_()
        self.pi = nn.Parameter(torch.tensor(linfo["pi"], dtype=torch.float32))
        # len_distribution = norm(linfo['mean'], linfo['std'])
        # self.pi = nn.Parameter(torch.tensor([len_distribution.pdf(i) for i in range(self.max_len)]))
        self.pi_noise_model = nn.Parameter(
            torch.tensor(linfo["pi"], dtype=torch.float32)
        )
        # self.pi_noise_model = nn.Parameter(torch.tensor([len_distribution.pdf(i) for i in range(self.max_len)]))
        self.pi.requires_grad_(False)
        self.pi_noise_model.requires_grad_(False)

    def trf_score(
        self, inputs: torch.Tensor, in_lens: torch.Tensor, targets: torch.Tensor
    ):
        # get the log prob of N sentences
        energy = self.calculate_energy(inputs, targets, in_lens)  # (B, )
        in_lens = torch.clamp(in_lens, max=self.pi.size(0) - 1)
        phi = -energy - self.zeta[in_lens]  # (B, )
        out = phi + torch.log(self.pi[in_lens])  # (B, )
        return out, phi

    def score(
        self,
        input_ids: torch.LongTensor,
        targets: torch.LongTensor,
        in_lens: torch.Tensor,
        *args
    ):
        if self.bert_tokenizer and input_ids[0][0] == 0:
            # the input sequence need to be processed:
            # input: 0[CLS]abcde[SEP] --> [CLS]abcde[SEP]
            # target: [CLS]abcde[SEP]0 --> abcde[SEP]0
            input_ids = input_ids[:, 1:]  # delete 0 in the head
            targets = targets[:, 1:]
            in_lens -= 1
        if self.noise_score:
            score = self.noisem_score(input_ids, in_lens, targets)
        else:
            score, _ = self.trf_score(input_ids, in_lens, targets)
        return score

    def gettrfdata(self, num: int = 10, turn_num: int = 100):
        """
        The text used to generate TRF using MIS.
        num is the number of channels used during generation
        (number of generated sentences),
        turn_num is the number of cycles run.
        """
        lendata = torch.multinomial(self.pi, num, True)
        n = 0
        # FIXME (huahuan): why here quantize the lendata to integer?
        maxlendata = int(max(lendata))
        # sentence slected in trf_data
        trfdata = torch.zeros(
            [num, maxlendata],
            device=next(self.noise_module[0].parameters()).device,
            dtype=torch.long,
        )
        # log prob of the slected sentence
        trfdata_log_pm = torch.zeros(
            [num],
            device=next(self.noise_module[0].parameters()).device,
            dtype=torch.long,
        )
        # log prob of the noise sentence
        trfdata_log_pn = torch.zeros(
            [num],
            device=next(self.noise_module[0].parameters()).device,
            dtype=torch.long,
        )
        for time in range(turn_num):  # MIS iteration
            noise = torch.zeros(
                [num, maxlendata],
                device=next(self.noise_module[0].parameters()).device,
                dtype=torch.long,
            )
            ones = torch.ones(
                [num],
                device=next(self.noise_module[0].parameters()).device,
                dtype=torch.long,
            )
            noise_next = torch.zeros(
                [num, 1],
                device=next(self.noise_module[0].parameters()).device,
                dtype=torch.long,
            )
            cache = None
            for i in range(maxlendata - 1):
                with torch.no_grad():
                    noise_out, cache = self.noise_module[0](
                        noise_next, cache=cache, input_lengths=ones
                    )
                noise_out = noise_out[:, 0, :]
                noise_distribution = F.softmax(noise_out, dim=-1)
                noise_next = torch.multinomial(
                    noise_distribution, 1, True
                )  # sampling by probablity
                noise[:, i + 1] = noise_next.squeeze(1)
            padding_mask = torch.arange(noise.size(1), device=noise.device)[
                None, :
            ] < lendata[:, None].to(noise.device)
            noise *= padding_mask
            tar = torch.cat(
                [noise[:, 1:maxlendata], noise[:, 0].unsqueeze(1)], dim=1
            ).unsqueeze(2)
            log_pm, phi = self.trf_score(noise, lendata, tar)
            log_pn = self.noisem_score(noise, lendata, tar)
            if time == 0:
                trfdata_log_pm = log_pm
                trfdata_log_pn = log_pn
                trfdata = noise
            else:
                p = -trfdata_log_pm + log_pm + trfdata_log_pn - log_pn
                p = p.exp()
                rand = torch.rand(
                    [num], device=next(self.noise_module[0].parameters()).device
                )
                for j in range(num):
                    # compute the rata of new sentence and old sentence, as the prob of replacing
                    if rand[j] < p[j]:
                        trfdata_log_pm[j] = log_pm[j]
                        trfdata_log_pn[j] = log_pn[j]
                        trfdata[j, :] = noise[j, :]
                        n += 1

    def getnoise(self, noise_num: int):
        # generate noise sentence，noise_num is the number of noise
        # generate the required sentence length with a priori probability
        lennoise = torch.multinomial(self.pi_noise_model, noise_num, True).to(
            next(self.noise_module[0].parameters()).device
        )
        maxlennoise = int(max(lennoise))
        noise = torch.zeros(
            [noise_num, maxlennoise],
            device=next(self.noise_module[0].parameters()).device,
            dtype=torch.long,
        )
        # When a token is generated in each round, the input length of each sentence is 1 (the rest are saved in the cache), and ones is the new input length
        ones = torch.ones(
            [noise_num],
            device=next(self.noise_module[0].parameters()).device,
            dtype=torch.long,
        )
        # predict next noise token
        if self.bert_tokenizer:
            # initialize the start token id with [CLS] id (101)
            noise_next = 101 * torch.ones(
                [noise_num, 1],
                device=next(self.noise_module[0].parameters()).device,
                dtype=torch.long,
            )
            noise[:, 0] = 101 * torch.ones(
                [noise_num],
                device=next(self.noise_module[0].parameters()).device,
                dtype=torch.long,
            )
        else:
            noise_next = torch.zeros(
                [noise_num, 1],
                device=next(self.noise_module[0].parameters()).device,
                dtype=torch.long,
            )
        extra_tokens = 2 if self.bert_tokenizer else 1
        generation_times = maxlennoise - extra_tokens
        noise_probs = torch.zeros(
            [noise_num, generation_times],
            device=next(self.noise_module[0].parameters()).device,
            dtype=torch.float32,
        )
        cache = None
        for i in range(generation_times):
            with torch.no_grad():
                if self.noise_type == "PretrainedTransformer":
                    noise_out, cache = self.noise_module[0](
                        noise_next, cache=cache, use_cache=True
                    )
                else:
                    noise_out, cache = self.noise_module[0](
                        src_ids=noise_next, cache=cache, input_lengths=ones
                    )
            noise_out = noise_out[:, 0, :]

            noise_distribution = F.softmax(noise_out, dim=-1)
            noise_next = torch.multinomial(noise_distribution, 1, True)
            probs = noise_distribution.gather(index=noise_next, dim=-1).squeeze()
            noise_probs[:, i] = probs
            noise[:, i + 1] = noise_next.squeeze(1)

        padding_mask = torch.arange(noise.size(1), device=noise.device)[
            None, :
        ] < lennoise[:, None].to(noise.device)
        noise *= padding_mask
        if self.bert_tokenizer:
            end_tokens = 102 * noise.new_ones(noise.shape)
            # add end tokens
            noise.scatter_(-1, (lennoise - 1).unsqueeze(-1), end_tokens)
        padding_mask_for_probs = torch.arange(noise_probs.size(1), device=noise.device)[
            None, :
        ] < (lennoise - extra_tokens)[:, None].to(noise.device)
        noise_probs = (torch.log(noise_probs) * padding_mask_for_probs).sum(
            dim=-1
        ) + torch.log(self.pi[lennoise])
        # noise sample for bert: [CLS]abcde[SEP], lennoise: 5+2, noise_probs: the probs of abcede
        # noise sample for others: 0abcde, lennoise: 5+1
        return noise, lennoise, noise_probs

    def cal_loss(self, inputs: torch.Tensor, energy_values, in_lens, targets):
        """
        input and target sample:
        For Bert: [CLS]abcde[SEP]  abcde[SEP]0
        For others: 0abcde  abcde0
        """

        data_sample_num = energy_values.shape[0]
        noise_sample_num = int(data_sample_num * self.noise_rate)

        if targets.dim() == 2:
            targets = targets.unsqueeze(2)

        if self.method == "nce":
            phi = -energy_values - self.zeta[in_lens]

            log_pm = phi + torch.log(self.pi[in_lens])

            log_pn = self.noisem_score(inputs, in_lens, targets)
            # ppl_data = torch.exp(-log_pm.sum()/in_lens.sum())
            with torch.no_grad():
                p1 = torch.sigmoid(math.log(self.noise_rate) - log_pm + log_pn)
            loss_data = -(p1 * phi).mean(dim=0)

            seqs, seqlens, log_pn = self.getnoise(noise_sample_num)
            seq_targets = seqs[:, 1:]  # (B, T-1)
            log_pm, phi = self.trf_score(seqs, seqlens, seq_targets)
            with torch.no_grad():
                p0 = torch.sigmoid(-math.log(self.noise_rate) + log_pm - log_pn)
            loss_noise = self.noise_rate * (p0 * phi).mean(dim=0)
            acc_data = (p1.data < 0.5).sum() / data_sample_num
            acc_noise = (p0.data < 0.5).sum() / noise_sample_num
            loss = loss_data + loss_noise
            return loss, {
                "train/loss_data": loss_data.detach(),
                "train/loss_noise": loss_noise.detach(),
                "train/acc_data": acc_data.detach(),
                "train/acc_noise": acc_noise.detach(),
            }

        elif self.method == "dnce":
            # noise in noise data
            noise_num_real = int(noise_sample_num / self.alpha)  # B2
            # number of extra noise need to use in data
            data_noise_num = int(data_sample_num * (1 - self.alpha) / self.alpha)  # B1

            phi = -energy_values - self.zeta[in_lens]
            log_pm = phi + torch.log(self.pi[in_lens])
            ppl_data = torch.exp(-log_pm.sum() / in_lens.sum())
            log_pn = self.noisem_score(inputs, in_lens, targets)
            log_prob_data_sum = log_pm.sum().detach()
            log_prob_noise_sum = log_pn.sum().detach()
            # for training noise model
            # Minimize KL divergence between p_d and p_n
            loss_noisem_ml = -log_pn.sum() / in_lens.sum()
            ppl_data_onnoise = torch.exp(loss_noisem_ml)

            # noise processing in mixed distribution
            if data_noise_num > 0:  # alpah<1
                seqs, seqlens, log_pn_noise_data = self.getnoise(data_noise_num)
                seq_targets = seqs[:, 1:]
                log_pm_noise_data, phi_noise_data = self.trf_score(
                    seqs, seqlens, seq_targets
                )
                # merge real data and noise into a mixed set
                log_pm = torch.cat([log_pm, log_pm_noise_data])
                log_pn = torch.cat([log_pn, log_pn_noise_data])
                phi = torch.cat([phi, phi_noise_data])

                helpalpha = torch.ones(
                    [int(data_sample_num / self.alpha)], device=inputs.device
                )
                # in binary classification, the probability corresponding to the mixed set is also replaced by the interpolation of model probability and noise probability
                log_pm = torch.logaddexp(
                    torch.log(self.alpha * helpalpha) + log_pm,
                    torch.log((1 - self.alpha) * helpalpha) + log_pn,
                )

            with torch.no_grad():
                p1 = torch.sigmoid(math.log(self.noise_rate) - log_pm + log_pn)

            # loss_data=-(p1*phi).mean(dim=0)
            loss_data = -torch.matmul(p1, phi / data_sample_num) * self.alpha
            loss_data_true = -torch.mean(torch.log(1 - p1 + self.episilon))

            # noise negative sample
            seqs2, seqlens2, log_pn_noise = self.getnoise(noise_num_real)
            ppl_noise_onnoise = torch.exp(-log_pn_noise.sum() / seqlens2.sum())
            seq_targets2 = seqs2[:, 1:]
            log_pm_noise, phi_noise = self.trf_score(seqs2, seqlens2, seq_targets2)
            if self.alpha < 1:
                helpalpha_noise = torch.ones([noise_num_real], device=inputs.device)
                log_pm_noise = torch.logaddexp(
                    torch.log(self.alpha * helpalpha_noise) + log_pm_noise,
                    torch.log((1 - self.alpha) * helpalpha_noise) + log_pn_noise,
                )

            ppl_noise = torch.exp(-log_pm_noise.sum() / seqlens2.sum())
            with torch.no_grad():
                p0 = torch.sigmoid(
                    log_pm_noise - log_pn_noise - math.log(self.noise_rate)
                )
            loss_noise = torch.matmul(p0, phi_noise / data_sample_num) * self.alpha
            loss_noise_true = -self.noise_rate * torch.mean(
                torch.log(1 - p0 + self.episilon)
            )
            # compute the prediction accuracy of all samples
            acc_data = sum(p1.data < 0.5) / int(p1.shape[0])
            acc_data_sample = sum(p1.data[:data_sample_num] < 0.5) / data_sample_num
            acc_data_noise = (
                sum(p1.data[data_sample_num:] < 0.5) / data_noise_num
                if data_noise_num > 0
                else torch.tensor(0)
            )
            acc_noise = sum(p0.data < 0.5) / int(p0.shape[0])
            loss = loss_data + loss_noise + loss_noisem_ml
            return loss, {
                "train/loss_data": loss_data.detach(),
                "train/loss_noise": loss_noise.detach(),
                "train/loss_noise_kl": loss_noisem_ml.detach(),
                "train/acc_data": acc_data.detach(),
                # "train/acc_data_sample": acc_data_sample.detach(),
                # "train/acc_data_noise": acc_data_noise.detach(),
                "train/acc_noise": acc_noise.detach(),
                "train/ppl_data": ppl_data.detach(),
                "train/ppl_trfM_noise": ppl_noise.detach(),
                "train/ppl_noiseM_data": ppl_data_onnoise.detach(),
                "train/ppl_noiseM_noise": ppl_noise_onnoise.detach(),
                "train/loss_data_true": loss_data_true.detach(),
                "train/loss_noise_true": loss_noise_true.detach(),
                "train/loss_true": loss_data_true.detach() + loss_noise_true.detach(),
                "train/log_prob_trf": log_prob_data_sum,
                "train/log_prob_noise": log_prob_noise_sum,
                "train/zeta_5": self.zeta[5].cpu().item(),
                "train/zeta_15": self.zeta[15].cpu().item(),
                "train/zeta_25": self.zeta[25].cpu().item(),
            }
        else:
            raise RuntimeError


class REBM(EBM):
    """
    Residula energy based model
    """

    def __init__(self, noise_mask_ratio: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.noise_mask_ratio = noise_mask_ratio

    def score(
        self,
        input_ids: torch.LongTensor,
        targets: torch.LongTensor,
        in_lens: torch.Tensor,
        *args
    ):
        if self.bert_tokenizer and input_ids[0][0] == 0:
            input_ids = input_ids[:, 1:]  # delete 0 in the head
            targets = targets[:, 1:]
            in_lens -= 1
        energy = self.calculate_energy(input_ids, targets, in_lens)
        noise_score = self.noisem_score(input_ids, in_lens, targets)  # log noise_prob
        score = noise_score.to(input_ids.device) - energy
        return score

    def get_batch_noise(self, inputs, in_lens, targets):
        if inputs.size(0) == 0:
            return inputs, in_lens, targets
        with torch.no_grad():
            max_len = inputs.size(1)
            probs = self.noise_mask_ratio * torch.ones([max_len], device=inputs.device)
            probs[0] = 0
            masked_indices = torch.bernoulli(probs).bool()
            noise = inputs.clone()
            noiselens = in_lens.clone()
            noisetargets = targets.clone()
            for i in range(max_len):
                if not masked_indices[i]:
                    continue
                noise_input = inputs[:, :i]
                noise_out, _ = self.noise_module[0](src_ids=noise_input)  # (B,T,V)
                noise_next = noise_out[:, -1, :].argmax(-1)  # (B,)
                noise[:, i] = noise_next
                noisetargets[:, i - 1] = noise_next
            padding_mask = torch.arange(max_len, device=inputs.device)[
                None, :
            ] < in_lens[:, None].to(inputs.device)
            noise *= padding_mask
        return noise, noiselens, noisetargets

    def get_masked_noise(self, inputs, in_lens, targets):
        frac_part = self.noise_rate - int(self.noise_rate)
        noise, noiselens, noisetargets = self.get_batch_noise(
            inputs[:frac_part, :], in_lens[:frac_part], targets[:frac_part, :]
        )
        for k in range(int(self.noise_rate)):
            noise_k, noiselens_k, noisetargets_k = self.get_batch_noise(
                inputs, in_lens, targets
            )
            noise = torch.cat((noise, noise_k), dim=0)
            noiselens = torch.cat((noiselens, noiselens_k), dim=0)
            noisetargets = torch.cat((noisetargets, noisetargets_k), dim=0)
        return noise, noiselens, noisetargets

    def cal_loss(self, inputs: torch.Tensor, energy_values, in_lens, targets):
        loss_data = -torch.mean(
            F.logsigmoid(-energy_values - math.log(self.noise_rate))
        )
        seqs, seqlens, seqtargets = self.get_masked_noise(inputs, in_lens, targets)
        noise_energy = self.calculate_energy(seqs, seqtargets, seqlens)
        loss_noise = -torch.mean(F.logsigmoid(noise_energy + math.log(self.noise_rate)))
        loss = loss_data + loss_noise
        with torch.no_grad():
            p0_data = F.sigmoid(-energy_values - math.log(self.noise_rate))
            p1_noise = F.sigmoid(noise_energy + math.log(self.noise_rate))
            acc_data = sum(p0_data.data > 0.5) / int(p0_data.shape[0])
            acc_noise = sum(p1_noise.data > 0.5) / int(p1_noise.shape[0])
        return loss, {
            "train/loss_data": loss_data.detach(),
            "train/loss_noise": loss_noise.detach(),
            "train/acc_data": acc_data.detach(),
            "train/acc_noise": acc_noise.detach(),
        }


class EBM_IS(EBM):
    # train EBM using importance sampling
    # we call the proposal model noise model in the code
    def __init__(
        self,
        method: str = "IS",  # IS/MIS/Gibbs,
        sampling_method: str = "sequential",  # parallel or sequential
        sampling_cache: bool = False,  # use the same sampling chains during the whole training process
        freeze_noise: bool = False,
        update_q_with_p: bool = False,  # update the proposal network q by minimize the KL divergence between q and the energy network p
        over_sample_rate: int = 1,  # sampling ratio compared to batch size in MCMC
        sample_token_num: int = 1,  # token numbers in one Gibbs sampling step
        store_sample_path: str = None,  # the path of samples stored
        sample_buffer_ratio: int = 1,  # the ratio of sample buffer size to batch size during gibbs sampling
        **kwargs
    ):
        super().__init__(method=method, **kwargs)

        # assign settings for EBM training
        self.freeze_noise = freeze_noise
        self.method = method
        self.update_q_with_p = update_q_with_p
        self.sampling_cache = sampling_cache
        self.sampling_method = sampling_method
        self.over_sample_rate = over_sample_rate
        self.sample_token_num = sample_token_num
        self.store_sample_path = store_sample_path
        self.sample_buffer_ratio = sample_buffer_ratio

        if freeze_noise:  # freeze noise model
            self.noise_model.requires_grad_(False)

        if self.method in ["MIS", "Gibbs"]:
            # we need to maintain a parallel sampling chain
            (
                self.samples,
                self.sample_lens,
                self.sample_tars,
                self.log_p_probs,
                self.log_q_probs,
            ) = (None, None, None, None, None)

        if self.store_sample_path is not None:
            self.fp = open(self.store_sample_path, "a")

    def MIS_step_parallel(
        self,
        init_samples,
        init_sample_lens,
        init_sample_tars,
        log_p_probs,
        log_q_probs,
        batch_size,
    ):
        """
        one step MIS on the basis of init_samples
        init_samples: (B, T)
        p_probs: (B, ), the probabilities of p_theta on init_samples
        q_probs: (B, ), the probabilities of q_phi (the proposal network) on init_samples
        """
        accept_rate = 0
        if init_samples is None:
            new_samples, new_lens, new_targets = self.getnoise(noise_num=batch_size)
            with torch.no_grad():
                self.noise_model.requires_grad_(False)
                log_pm = -self.calculate_energy(
                    new_samples, new_targets, new_lens
                )  # unnormalized
                log_pn = self.noisem_score(new_samples, new_lens, new_targets)
                self.noise_model.requires_grad_(True)
            log_p_probs = log_pm
            log_q_probs = log_pn
            init_samples = new_samples
            init_sample_lens = new_lens
            init_sample_tars = new_targets
        else:
            new_samples, new_lens, new_targets = self.getnoise(noise_num=batch_size)
            with torch.no_grad():
                self.noise_model.requires_grad_(False)
                log_pm = -self.calculate_energy(
                    new_samples, new_targets, new_lens
                )  # unnormalized
                log_pn = self.noisem_score(new_samples, new_lens, new_targets)
                self.noise_model.requires_grad_(True)

            p = -log_p_probs + log_pm + log_q_probs - log_pn
            p = p.exp()
            rand = torch.rand(
                [batch_size], device=next(self.noise_model.parameters()).device
            )
            for j in range(batch_size):
                # compute the rata of new sentence and old sentence, as the prob of replacing
                if rand[j] < p[j]:
                    log_p_probs[j] = log_pm[j]
                    log_q_probs[j] = log_pn[j]
                    init_samples[j, :] = new_samples[j, :]
                    init_sample_lens[j] = new_lens[j]
                    init_sample_tars[j, :] = new_targets[j, :]
                    accept_rate += 1
        accept_rate /= batch_size
        return (
            init_samples,
            init_sample_lens,
            init_sample_tars,
            log_p_probs,
            log_q_probs,
            accept_rate,
        )

    def MIS_step_sequential(
        self,
        last_sample,
        last_sample_len,
        last_sample_tar,
        log_p_prob,
        log_q_prob,
        batch_size,
    ):
        """
        one step MIS on the basis of the last sample of previous iteration
        """
        # generate proposals and calculate log probs
        accept_rate = 0
        accept_count = 0
        new_samples, new_lens, new_targets = self.getnoise(
            noise_num=batch_size * self.over_sample_rate
        )
        with torch.no_grad():
            self.noise_model.requires_grad_(False)
            log_pm = -self.calculate_energy(
                new_samples, new_targets, new_lens
            )  # unnormalized
            log_pn = self.noisem_score(new_samples, new_lens, new_targets)
            self.noise_model.requires_grad_(True)
        # initialize the final samples
        final_samples, final_sample_lens, final_sample_tars = (
            deepcopy(new_samples),
            deepcopy(new_lens),
            deepcopy(new_targets),
        )
        final_log_p, final_log_q = deepcopy(log_pm), deepcopy(log_pn)
        if last_sample is not None:
            # put it in final_samples for convenience
            final_samples[-1, :] = last_sample
            final_sample_lens[-1] = last_sample_len
            final_sample_tars[-1, :] = last_sample_tar
            final_log_p[-1] = log_p_prob
            final_log_q[-1] = log_q_prob

        for j in range(batch_size * self.over_sample_rate):
            accept_flag = False
            if j == 0:
                if last_sample is None:  # directly accept the first sample
                    accept_flag = True
                else:  # compared with the last sample of previous iteration
                    log_p = log_pm[j] - log_pn[j] + final_log_q[-1] - final_log_p[-1]
                    p = log_p.exp()
                    rand = torch.rand(
                        1, device=next(self.noise_model.parameters()).device
                    )
                    accept_flag = p > rand
            else:
                log_p = log_pm[j] - log_pn[j] + final_log_q[j - 1] - final_log_p[j - 1]
                p = log_p.exp()
                rand = torch.rand(1, device=next(self.noise_model.parameters()).device)
                accept_flag = p > rand

            if accept_flag:
                final_log_p[j] = log_pm[j]
                final_log_q[j] = log_pn[j]
                final_samples[j, :] = new_samples[j, :]
                final_sample_lens[j] = new_lens[j]
                final_sample_tars[j, :] = new_targets[j, :]
                accept_count += 1
            else:
                final_log_p[j] = final_log_p[j - 1]
                final_log_q[j] = final_log_q[j - 1]
                final_samples[j, :] = final_samples[j - 1, :]
                final_sample_lens[j] = final_sample_lens[j - 1]
                final_sample_tars[j, :] = final_sample_tars[j - 1, :]
            if j % self.over_sample_rate == 0:
                accept_rate += accept_count > 0
                accept_count = 0
        accept_rate /= batch_size
        if self.over_sample_rate > 1:  # recover the size
            final_samples = final_samples[:: self.over_sample_rate, :]
            final_sample_lens = final_sample_lens[:: self.over_sample_rate]
            final_sample_tars = final_sample_tars[:: self.over_sample_rate, :]
            final_log_p = final_log_p[:: self.over_sample_rate]
            final_log_q = final_log_q[:: self.over_sample_rate]
        return (
            final_samples,
            final_sample_lens,
            final_sample_tars,
            final_log_p,
            final_log_q,
            accept_rate,
        )

    def MH_within_Gibbs(self, init_batch, init_batch_len, granularity=1):
        """
        MH within Gibbs sampling
        At present, this method is only applicable to the case that the energy model and proposal model are both of BERT architecture
        granularity: token numbers in one Gibbs step
        """
        accept_rate = 0
        total = 0
        with torch.no_grad():
            noise = init_batch.clone()
            right_bound = init_batch.shape[1] - 1
            for t in range(1, right_bound, granularity):
                # mask the t-granularity tokens
                masked_inputs = noise.clone()
                B = init_batch.size(0)  # batch size
                G = min(t + granularity, right_bound) - t  # granularity
                masked_inputs[:, t : t + G] = 103 * torch.ones(
                    [B, G], device=init_batch.device, dtype=torch.long
                )
                # sampling new tokens from proposal network (noise_model)
                outputs = self.noise_model(
                    masked_inputs, None, input_lengths=init_batch_len
                )
                noise_distribution = F.softmax(
                    outputs.logits[:, t : t + G, :], dim=-1
                )  # (B, G, V)
                noise_distribution = noise_distribution.view(
                    -1, noise_distribution.size(-1)
                )  # B*G, V
                noise_t = torch.multinomial(noise_distribution, 1, True)  # (B*G, 1)
                noise_t = noise_t.view(B, G, 1)  # (B,G,1)
                masked_inputs[:, t : t + G] = noise_t.squeeze(-1)
                noise_distribution = noise_distribution.view(B, G, -1)  # (B,G,V)
                log_q2 = torch.log(
                    noise_distribution.gather(index=noise_t, dim=-1).squeeze(-1)
                )  # (B,G)
                log_q1 = torch.log(
                    noise_distribution.gather(
                        index=noise[:, t : t + G].unsqueeze(-1), dim=-1
                    ).squeeze(-1)
                )  # (B,G)
                log_p1 = -self.calculate_energy(noise, None, init_batch_len)  # (B,)
                log_p2 = -self.calculate_energy(
                    masked_inputs, None, init_batch_len
                )  # (B,)
                log_p = log_p2 - log_q2.sum(-1) + log_q1.sum(-1) - log_p1
                for j in range(B):
                    p = log_p[j].exp()
                    rand = torch.rand(1, device=p.device)
                    total += 1
                    if rand < p and t < init_batch_len[j] - 1:
                        token_num = min(t + G, init_batch_len[j] - 1) - t
                        noise[j, t : t + token_num] = noise_t[j, :token_num, :].squeeze(
                            -1
                        )
                        accept_rate += 1
        accept_rate /= total
        padding_mask = torch.arange(init_batch.size(1), device=init_batch.device)[
            None, :
        ] < init_batch_len[:, None].to(init_batch.device)
        change_rate = (
            (noise != init_batch) * padding_mask
        ).sum() / init_batch_len.sum()
        return noise, accept_rate, change_rate

    def Gibbs(self, init_batch, init_batch_len, granularity=1):
        """
        Gibbs sampling
        At present, this method is only applicable to the case that the energy model and proposal model are both of BERT architecture
        granularity: token numbers in one Gibbs step
        """
        accept_rate = 0
        with torch.no_grad():
            noise = init_batch.clone()
            right_bound = init_batch.shape[1] - 1
            for t in range(1, right_bound, granularity):
                # mask the t-granularity tokens
                B = init_batch.size(0)  # batch size
                G = min(t + granularity, right_bound) - t  # granularity
                noise[:, t : t + G] = 103 * torch.ones(
                    [B, G], device=init_batch.device, dtype=torch.long
                )
                # sampling new tokens from proposal network (noise_model)
                outputs = self.noise_model(noise, None, input_lengths=init_batch_len)
                noise_distribution = F.softmax(
                    outputs.logits[:, t : t + G, :], dim=-1
                )  # (B, G, V)
                noise_distribution = noise_distribution.view(
                    -1, noise_distribution.size(-1)
                )  # B*G, V
                noise_t = torch.multinomial(noise_distribution, 1, True)  # (B*G, 1)
                noise_t = noise_t.view(B, G, 1)  # (B,G,1)
                noise[:, t : t + G] = noise_t.squeeze(-1)

        padding_mask = torch.arange(init_batch.size(1), device=init_batch.device)[
            None, :
        ] < init_batch_len[:, None].to(init_batch.device)
        change_rate = (
            (noise != init_batch) * padding_mask
        ).sum() / init_batch_len.sum()
        return noise, accept_rate, change_rate

    def getnoise(self, noise_num, maxlennoise=40):
        """
        use auto-regressive model to generate noise samples
        this function is the same as get_noise() in EBM, but we need to use two require_grad functions
        to prevent a strange bug in distributed training in pytorch
        """
        with torch.no_grad():
            self.noise_model.requires_grad_(False)
            noise = torch.zeros(
                [noise_num, maxlennoise],
                device=next(self.noise_model.parameters()).device,
                dtype=torch.long,
            )
            ones = torch.ones(
                [noise_num],
                device=next(self.noise_model.parameters()).device,
                dtype=torch.long,
            )
            if self.bert_tokenizer:
                # initialize the start token id with [CLS] id (101)
                noise_next = 101 * torch.ones(
                    [noise_num, 1],
                    device=next(self.noise_model.parameters()).device,
                    dtype=torch.long,
                )
                noise[:, 0] = 101 * torch.ones(
                    [noise_num],
                    device=next(self.noise_model.parameters()).device,
                    dtype=torch.long,
                )
            else:
                noise_next = torch.zeros(
                    [noise_num, 1],
                    device=next(self.noise_model.parameters()).device,
                    dtype=torch.long,
                )
            cache = None
            is_end = torch.zeros([noise_num], dtype=torch.bool, device=noise.device)
            lennoise = torch.ones([noise_num], dtype=torch.long, device=noise.device)
            end_mark = 102 if self.bert_tokenizer else 0
            for i in range(maxlennoise - 1):
                if self.noise_cls == "PretrainedTransformer":
                    noise_out, cache = self.noise_model(
                        noise_next, cache=cache, use_cache=True
                    )
                else:
                    noise_out, cache = self.noise_model(
                        src_ids=noise_next, cache=cache, input_lengths=ones
                    )
                noise_out = noise_out[:, -1, :]  # (B,V)
                noise_distribution = F.softmax(noise_out, dim=-1)
                noise_next = torch.multinomial(noise_distribution, 1, True)  # (B, 1)
                noise[:, i + 1] = noise_next.squeeze(1)
                lennoise += ones * (~is_end)
                is_end |= noise_next.squeeze(-1) == end_mark
                if all(is_end):
                    break

            padding_mask = torch.arange(noise.size(1), device=noise.device)[
                None, :
            ] < lennoise[:, None].to(noise.device)
            noise *= padding_mask
            targets = torch.cat(
                (
                    noise[:, 1:],
                    torch.zeros([noise_num, 1], device=noise.device, dtype=torch.long),
                ),
                dim=1,
            )
            self.noise_model.requires_grad_(True)
        return noise, lennoise, targets

    def getnoise_mlm(self, init_batch, init_batch_len, mode="PLL"):
        """
        use masked language model to generate noise models.
        mode: PLL(pseudolikelihood) or EBM (energy based model)
        init_batch: (B,T)
        """
        with torch.no_grad():
            if mode == "PLL":
                # mask each token and obtain its PLL for sampling
                noise = init_batch.new_zeros(init_batch.shape)
                noise[:, 0] = init_batch[:, 0]
                noise[:, -1] = init_batch[:, -1]
                init_log_probs = torch.zeros(
                    [init_batch.size(0)], device=init_batch.device, dtype=torch.float32
                )
                new_log_probs = torch.zeros(
                    [init_batch.size(0)], device=init_batch.device, dtype=torch.float32
                )
                for t in range(1, init_batch.shape[1] - 1):
                    masked_inputs = init_batch.clone()
                    # mask token id: 103
                    masked_inputs[:, t] = 103 * torch.ones(
                        [init_batch.shape[0]],
                        device=init_batch.device,
                        dtype=torch.long,
                    )
                    outputs = self.noise_model(
                        masked_inputs, input_lengths=init_batch_len
                    )
                    assert "logits" in outputs, "The output has no attribute logits"
                    noise_distribution = F.softmax(
                        outputs.logits[:, t, :], dim=-1
                    )  # (B, V)
                    noise_t = torch.multinomial(noise_distribution, 1, True)  # (B, 1)
                    noise[:, t] = noise_t.squeeze(1)
                    new_log_probs += torch.log(
                        noise_distribution.gather(index=noise_t, dim=-1).squeeze(-1)
                    )
                    init_log_probs += torch.log(
                        noise_distribution.gather(
                            index=init_batch[:, t].unsqueeze(1), dim=-1
                        ).squeeze(-1)
                    )
            else:
                pass
        return noise, init_log_probs, new_log_probs

    def cal_loss(self, inputs: torch.Tensor, energy_values, in_lens, targets):
        data_sample_num = energy_values.shape[0]
        noise_sample_num = int(data_sample_num * self.noise_rate)

        if targets.dim() == 2:
            targets = targets.unsqueeze(2)
        loss_data = torch.mean(energy_values)
        # calculate loss_sampling
        if self.method == "IS":
            noise, noiselens, noisetars = self.getnoise(noise_sample_num)
            energy_noise = self.calculate_energy(noise, noisetars, noiselens)
            with torch.no_grad():
                self.noise_model.requires_grad_(False)
                log_p_theta = -energy_noise
                log_q_phi = self.noisem_score(
                    noise, noiselens, noisetars
                )  # log probabilities of sentences
                log_weight = log_p_theta - log_q_phi
                weight_norm = F.softmax(log_weight, dim=-1)
                self.noise_model.requires_grad_(True)
            loss_sampling = torch.sum(weight_norm * energy_noise)
        elif self.method == "MIS":
            if self.sampling_method == "sequential":
                if self.sampling_cache and self.samples is not None:
                    (
                        self.samples,
                        self.sample_lens,
                        self.sample_tars,
                        self.log_p_probs,
                        self.log_q_probs,
                        accept_rate,
                    ) = self.MIS_step_sequential(
                        self.samples[-1, :],
                        self.sample_lens[-1],
                        self.sample_tars[-1, :],
                        self.log_p_probs[-1],
                        self.log_q_probs[-1],
                        noise_sample_num,
                    )
                else:
                    (
                        self.samples,
                        self.sample_lens,
                        self.sample_tars,
                        self.log_p_probs,
                        self.log_q_probs,
                        accept_rate,
                    ) = self.MIS_step_sequential(
                        None, None, None, None, None, noise_sample_num
                    )
            else:
                (
                    self.samples,
                    self.sample_lens,
                    self.sample_tars,
                    self.log_p_probs,
                    self.log_q_probs,
                    accept_rate,
                ) = self.MIS_step_parallel(
                    self.samples,
                    self.sample_lens,
                    self.sample_tars,
                    self.log_p_probs,
                    self.log_q_probs,
                    noise_sample_num,
                )
            energy_noise = self.calculate_energy(
                self.samples, self.sample_tars, self.sample_lens
            )
            loss_sampling = torch.mean(energy_noise)
        elif self.method == "Gibbs":
            if self.sampling_cache:
                if self.samples is None:  # initialize the sampling buffer
                    self.samples, self.sample_lens = inputs.clone(), in_lens.clone()
                elif self.samples.size(0) < self.sample_buffer_ratio * inputs.size(0):
                    merged_samples = torch.zeros(
                        (
                            self.samples.size(0) + inputs.size(0),
                            max(self.samples.size(1), inputs.size(1)),
                        ),
                        dtype=torch.long,
                        device=inputs.device,
                    )
                    merged_samples[
                        : self.samples.size(0), : self.samples.size(1)
                    ] = self.samples
                    merged_samples[self.samples.size(0) :, : inputs.size(1)] = inputs
                    self.sample_lens = torch.cat((self.sample_lens, in_lens), dim=0)
                    self.samples = merged_samples

                # sampling a batch from the buffer uniformly
                sampling_probs = torch.ones(self.samples.size(0)) / self.samples.size(0)
                init_sample_ids = torch.multinomial(sampling_probs, inputs.size(0))
                init_samples = self.samples[init_sample_ids]
                init_sample_lens = self.sample_lens[init_sample_ids]
                new_samples, accept_rate, change_rate = self.MH_within_Gibbs(
                    init_samples, init_sample_lens, self.sample_token_num
                )
                energy_noise = self.calculate_energy(
                    new_samples, None, init_sample_lens
                )
                self.samples[init_sample_ids] = new_samples
            else:
                noise, accept_rate, change_rate = self.MH_within_Gibbs(
                    inputs, in_lens, self.sample_token_num
                )
                energy_noise = self.calculate_energy(noise, targets, in_lens)
            loss_sampling = torch.mean(energy_noise)
        # calculate loss_noisem_ml
        if not self.freeze_noise:  # update the proposal network during training
            if self.update_q_with_p:
                # minimize the KL divergence between q_phi and p_theta
                log_q_phi_samples = self.noisem_score(
                    self.samples, self.sample_lens, self.sample_tars
                )
                loss_noisem_ml = -torch.mean(log_q_phi_samples)
            else:
                # directly minimize the KL divergence between q_phi and data
                log_q_phi_data = self.noisem_score(inputs, in_lens, targets)
                loss_noisem_ml = -torch.mean(log_q_phi_data)
        else:
            loss_noisem_ml = 0

        # store samples
        if next(self.udlying_nn.parameters()).device.index == 0:
            if self.store_sample_path is not None and self.samples is not None:
                content = "\n".join(
                    [
                        self.tokenizer.decode(
                            self.samples[k, : self.sample_lens[k]].cpu().tolist()
                        )
                        for k in range(10)
                    ]
                )
                self.fp.write(content + "\n")

        loss = loss_data - loss_sampling + loss_noisem_ml
        metrics = {
            "train/loss_data": loss_data.detach(),
            "train/loss_sampling": loss_sampling.detach(),
            "train/loss_kl": loss_noisem_ml,
        }
        if self.method in ["MIS", "Gibbs"]:
            metrics.update({"train/accept_rate": accept_rate})
        if self.method == "Gibbs":
            metrics.update({"train/change_rate": change_rate})
        return loss, metrics

        return self.calculate_energy(inputs, targets, input_lengths)
