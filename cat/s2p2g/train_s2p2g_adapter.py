# Copyright 2024 Tsinghua University
# Apache 2.0.
# Author: Sardar (sar_dar@163.com)

"""Top interface of RAG Sequence training.
    S2P: CTC based speech to phone model.
    P2G: AED based phone to BPE model.

"""

__all__ = ["S2P2GTaskModel", "build_model", "_parser", "main"]



import torch.distributed
from ..shared.manager import Manager
from ..shared import coreutils
from ..shared import encoder as model_zoo
from ..shared.data import S2P2GAdapterTrainDataset,S2P2GAdapterTrainCollate
from ..shared.tokenizer import load




import argparse
import Levenshtein
from collections import OrderedDict
from typing import *
# import ctc_align



from ctcdecode import CTCBeamDecoder

from tqdm import tqdm
import torch
import torch.nn as nn

import torch.distributed as dist


from transformers import AutoTokenizer

from transformers import Qwen3ForCausalLM

from .config_utils import PromptSpec, resolve_prompt_spec

DEFAULT_PROMPT_TEMPLATE = (
    "<|im_start|>user\n"
    "Transcribe speech to text. Output the transcription directly without redundant content. "
    "Ensure that the output is not duplicated.\n"
    "{s}<|im_end|><|im_start|>assistant\n{y}"
)


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace, **mkwargs):

    if args.rank == 0:
        print(f'dist_backend: {args.dist_backend}')
        print(f'args.dist_url: {args.dist_url}')
        print(f'world_size: {args.world_size}')
        print(f'ngpus_per_node: {ngpus_per_node}')
        print(f'rank: {args.rank}')

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
        mkwargs["T_dataset"] = S2P2GAdapterTrainDataset

    if "collate_fn" not in mkwargs:
        mkwargs["collate_fn"] = S2P2GAdapterTrainCollate(flatten_target=False)

    if "func_build_model" not in mkwargs:
        mkwargs["func_build_model"] = build_model

    if (
        "func_eval" not in mkwargs
        and hasattr(args, "eval_error_rate")
        and args.eval_error_rate
    ):
        mkwargs["func_eval"] = custom_evaluate


    
    mkwargs["args"] = args
    

    manager = Manager(**mkwargs)

    if args.ld is None:
        tr_dataset = manager.trainloader.dl.dataset
        coreutils.distprint(
            f"  total {tr_dataset.__len__()} utterances are used in training.", args.gpu
        )

    # training
    manager.run(args)



class LinearProjector(nn.Module):
    def __init__(self, speech_embedding_dim,projector_hidden_dim,llm_embedding_dim,k=1,freeze=False):
        super().__init__()
        self.k = k #做几次下采样
        self.speech_embedding_dim = speech_embedding_dim
        self.llm_embedding_dim = llm_embedding_dim
        self.projector_hidden_dim = projector_hidden_dim
        self.linear1 = nn.Linear(self.speech_embedding_dim * self.k, self.projector_hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.projector_hidden_dim, llm_embedding_dim)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x,x_len):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x,x_len//self.k



class S2P2GAdapterModel(nn.Module):
    def __init__(
        self,
        s2p_model: model_zoo.AbsEncoder,
        p2g_model,
        llm_tokenizer = None,
        prompt:str = "",
        prompt_template: str = "",
        prompt_spec: Optional[PromptSpec] = None,
        projector_config=None
    ):  
        """
        Args:
            s2p_model: 
            phn_decoder: 
            p2g_model: 
            llm_tokenizer: 
            s2p_tknz: 
            p_prompt: 
            train_config:
            phn_as_special_token: 
        """
        super().__init__()

        self.s2p_model = s2p_model
        self.p2g_model = p2g_model

        self.projector = LinearProjector(**projector_config)
                
        self.llm_tokenizer = llm_tokenizer
        
        if self.llm_tokenizer.pad_token_id is None:
            self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id

        self.p2g_model.llm_model.config.eos_token_id = llm_tokenizer.eos_token_id
        self.p2g_model.llm_model.config.pad_token_id = llm_tokenizer.pad_token_id

        if prompt_spec is None:
            prompt_cfg = {"prompt_config": {"prompt_template": prompt_template}} if prompt_template else {}
            if prompt and not prompt_template:
                # ``prompt`` is a legacy constructor argument. Keep old callers
                # that passed a full ``{s}/{y}`` template working, while the
                # JSON-facing ``prompt_config.prompt`` remains a strict prefix.
                prompt_cfg["prompt"] = prompt
            prompt_spec = resolve_prompt_spec(prompt_cfg, default_prompt=DEFAULT_PROMPT_TEMPLATE)
        self.prompt_spec = prompt_spec
        self.prompt_mode = prompt_spec.mode
        self.prompt_template = prompt_spec.prompt_template
        self.prefix_prompt = prompt_spec.prefix_prompt
        self.postfix_prompt = prompt_spec.postfix_prompt
        self.prompt = prompt_spec.prompt
        

    def forward(self,x,x_len,y_text,y_len):

       
        _,speech_embedding_len,speech_embedding = self.s2p_model(x,x_len,return_last_hidden_state=True)


        speech_embedding_len = speech_embedding_len.to(torch.int32)

        speech_embedding = self.mask_pad(speech_embedding,speech_embedding_len)

        speech_align_embedding,speech_align_len = self.projector(speech_embedding,speech_embedding_len)

        batch_input_output = self.process_input_output(speech_align_embedding,speech_align_len,y_text)

        llm_output = self.p2g_model.llm_model(inputs_embeds=batch_input_output["input_embeds"],attention_mask=batch_input_output["attention_mask"],labels=batch_input_output["labels"])

        return llm_output.loss

    def _embed_prompt_text(self, text: str, device: torch.device) -> torch.Tensor:
        embed_tokens = self.p2g_model.llm_model.model.embed_tokens
        ids = self.llm_tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            return embed_tokens.weight.new_zeros((0, embed_tokens.weight.size(1))).to(
                device=device
            )
        return embed_tokens(torch.tensor(ids, device=device, dtype=torch.long))

    def _embed_prompt_parts(
        self,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._embed_prompt_text(self.prefix_prompt, device),
            self._embed_prompt_text(self.postfix_prompt, device),
        )
    

    '''
    def process_input_output(
    self,
    batch_speech_embedding,
    batch_speech_embedding_length,
    batch_y,
    ):
        """
        prefix_prompt + speech + postfix_prompt + y + eos_token_id
        """

        eos_id = self.llm_tokenizer.eos_token_id
        device = batch_speech_embedding.device

        # --------------------------------------------------------
        #  prefix / postfix prompt → ids → embeddings
        # --------------------------------------------------------
        prefix_ids = self.llm_tokenizer.encode(
            self.prefix_prompt, add_special_tokens=False
        )
        postfix_ids = self.llm_tokenizer.encode(
            self.postfix_prompt, add_special_tokens=False
        )

        prefix_emb = self.p2g_model.llm_model.model.embed_tokens(
            torch.tensor(prefix_ids, device=device, dtype=torch.int32)
        )
        postfix_emb = self.p2g_model.llm_model.model.embed_tokens(
            torch.tensor(postfix_ids, device=device, dtype=torch.int32)
        )

        sample_list = []

        # --------------------------------------------------------
        #  遍历 batch
        # --------------------------------------------------------
        for speech_emb, speech_len, y in zip(
            batch_speech_embedding,
            batch_speech_embedding_length,
            batch_y,
        ):

            # 有效 speech embedding
            speech_emb = speech_emb[:speech_len]  # [T, D]

            # 输出 y → ids → embeddings
            y_ids = self.llm_tokenizer.encode(y, add_special_tokens=False)
            y_ids = torch.tensor(y_ids, dtype=torch.long, device=device)

            y_with_eos = torch.cat(
                [y_ids, torch.tensor([eos_id], device=device)],
                dim=0,
            )  # [L_out + 1]

            y_emb = self.p2g_model.llm_model.model.embed_tokens(y_with_eos)

            # --------------------------------------------------------
            #  拼接顺序：
            #  prefix + speech + postfix + y + eos
            # --------------------------------------------------------
            combined_emb = torch.cat(
                [prefix_emb, speech_emb, postfix_emb, y_emb],
                dim=0,
            )

            # attention mask
            attn_mask = torch.ones(
                combined_emb.size(0),
                dtype=torch.long,
                device=device
            )

            # --------------------------------------------------------
            #  labels
            #  prefix + speech + postfix 不训练，标记为 -100
            # --------------------------------------------------------
            prefix_len = prefix_emb.size(0)
            speech_len = speech_emb.size(0)
            postfix_len = postfix_emb.size(0)

            no_loss_len = prefix_len + speech_len + postfix_len

            labels = torch.cat(
                [
                    torch.full(
                        (no_loss_len,),
                        -100,
                        dtype=torch.long,
                        device=device
                    ),
                    y_with_eos,   # 需要训练的部分
                ],
                dim=0,
            )

            # 保存每个样本
            sample_list.append(
                dict(
                    input_embeds=combined_emb,
                    attention_mask=attn_mask,
                    labels=labels,
                    seq_len=combined_emb.size(0)
                )
            )

        # ==========================================================
        #                  Batch Padding
        # ==========================================================

        batch_size = len(sample_list)
        max_len = max(s["seq_len"] for s in sample_list)
        embed_dim = sample_list[0]["input_embeds"].size(-1)

        batch_embeds = torch.zeros(
            (batch_size, max_len, embed_dim),
            dtype=torch.float32,
            device=device,
        )
        batch_labels = torch.full(
            (batch_size, max_len),
            -100,
            dtype=torch.long,
            device=device,
        )
        batch_attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )

        # copy each sample into batch
        for i, s in enumerate(sample_list):
            L = s["seq_len"]
            batch_embeds[i, :L] = s["input_embeds"]
            batch_labels[i, :L] = s["labels"]
            batch_attention_mask[i, :L] = 1

        return dict(
            input_embeds=batch_embeds,          # [B, L, D]
            attention_mask=batch_attention_mask,
            labels=batch_labels,
        )

    def build_generate_inputs_left_align(
        self,
        batch_speech_embedding,
        batch_speech_embedding_length,
        ):
        """
        左对齐版 generate 输入构建：
        final = [PAD ... PAD | prefix_prompt | speech | postfix_prompt]

        注意：generate 阶段不包含 y
        """

        device = batch_speech_embedding.device

        # ===========================================================
        # prefix & postfix prompt embedding
        # ===========================================================
        prefix_ids = self.llm_tokenizer.encode(
            self.prefix_prompt, add_special_tokens=False
        )
        postfix_ids = self.llm_tokenizer.encode(
            self.postfix_prompt, add_special_tokens=False
        )

        prefix_emb = self.p2g_model.llm_model.model.embed_tokens(
            torch.tensor(prefix_ids, device=device, dtype=torch.int32)
        )  # [L_pre, D]

        postfix_emb = self.p2g_model.llm_model.model.embed_tokens(
            torch.tensor(postfix_ids, device=device, dtype=torch.int32)
        )  # [L_post, D]

        L_pre = prefix_emb.size(0)
        L_post = postfix_emb.size(0)

        # ===========================================================
        # collect each sample
        # ===========================================================
        sample_list = []

        for speech_emb, speech_len in zip(
            batch_speech_embedding,
            batch_speech_embedding_length,
        ):
            speech_emb = speech_emb[:speech_len]   # [T, D]
            L_speech = speech_emb.size(0)

            total_len = L_pre + L_speech + L_post

            sample_list.append(
                dict(
                    prefix_emb=prefix_emb,
                    speech_emb=speech_emb,
                    postfix_emb=postfix_emb,
                    seq_len=total_len
                )
            )

        # ===========================================================
        # batch padding
        # ===========================================================
        batch_size = len(sample_list)
        max_len = max(s["seq_len"] for s in sample_list)
        embed_dim = sample_list[0]["speech_emb"].size(-1)

        # containers
        batch_embeds = torch.zeros(
            (batch_size, max_len, embed_dim),
            dtype=torch.float32,
            device=device,
        )
        batch_attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )

        # ===========================================================
        # left padding
        # format: [PAD ... PAD | prefix | speech | postfix]
        # ===========================================================
        for i, s in enumerate(sample_list):
            L = s["seq_len"]
            pad_len = max_len - L  # left padding size

            merged = torch.cat(
                [s["prefix_emb"], s["speech_emb"], s["postfix_emb"]],
                dim=0
            )

            batch_embeds[i, pad_len : pad_len + L] = merged
            batch_attention_mask[i, pad_len : pad_len + L] = 1

        return dict(
            input_embeds=batch_embeds,              # [B, L, D]
            attention_mask=batch_attention_mask,    # [B, L]
        )
    '''






    # '''
    def process_input_output(
    self,
    batch_speech_embedding,
    batch_speech_embedding_length,
    batch_y,
        ):
        """
        prefix_prompt + speech + postfix_prompt + y + eos_token_id
        """

        eos_id = self.llm_tokenizer.eos_token_id
        device = batch_speech_embedding.device

        prefix_emb, postfix_emb = self._embed_prompt_parts(device)
        
        sample_list = []



        # -------- 遍历 batch -------- #
        for speech_emb, speech_len, y in zip(
            batch_speech_embedding,
            batch_speech_embedding_length,
            batch_y,
        ):

            y_ids = self.llm_tokenizer.encode(y, add_special_tokens=False)

            # print(speech_len)

            # 有效 speech embedding
            speech_emb = speech_emb[:speech_len]  # [T, D]

            # 输出 tokens
            y_ids = torch.tensor(y_ids, dtype=torch.long, device=device)

            # -------- 添加 EOS token -------- #
            y_with_eos = torch.cat(
                [y_ids, torch.tensor([eos_id], device=device)],
                dim=0
            )  # [L_out + 1]

            # 输出 embedding
            y_emb = self.p2g_model.llm_model.model.embed_tokens(y_with_eos)  # [L_out+1, D]

            # -------- 拼接最终 embedding -------- #
            combined_emb = torch.cat(
                [prefix_emb, speech_emb, postfix_emb, y_emb],
                dim=0
            )  # [L_total, D]

            # -------- attention mask -------- #
            attn_mask = torch.ones(
                combined_emb.size(0),
                dtype=torch.long,
                device=device
            )

            # -------- labels -------- #
            prompt_len = prefix_emb.size(0) + postfix_emb.size(0)
            speech_len = speech_emb.size(0)
            

            # prefix + speech + postfix parts do not contribute LM loss.
            no_loss_len = prompt_len + speech_len 

            labels = torch.cat(
                [
                    torch.full(
                        (no_loss_len,), -100,
                        dtype=torch.long, device=device
                    ),
                    y_with_eos  # 参与训练
                ],
                dim=0
            )

            # 存储该样本
            sample_list.append(
                dict(
                    input_embeds=combined_emb,
                    attention_mask=attn_mask,
                    labels=labels,
                    seq_len=combined_emb.size(0)
                )
            )
        # ==========================================================
        #                  Batch Padding 逻辑
        # ==========================================================

        batch_size = len(sample_list)
        max_len = max(s["seq_len"] for s in sample_list)
        embed_dim = sample_list[0]["input_embeds"].size(-1)

        # -------- 创建 batch tensor -------- #
        batch_embeds = torch.zeros(
            (batch_size, max_len, embed_dim),
            dtype=torch.float32,
            device=device,
        )


        batch_labels = torch.full(
            (batch_size, max_len),
            -100,
            dtype=torch.long,
            device=device,
        )
        batch_attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )

        # -------- 将每个样本 copy 到 batch tensor -------- #
        for i, s in enumerate(sample_list):
            L = s["seq_len"]
            batch_embeds[i, :L] = s["input_embeds"]
            batch_labels[i, :L] = s["labels"]
            batch_attention_mask[i, :L] = 1

        # 最终输出
        return dict(
            input_embeds=batch_embeds,         # [B, L, D]
            attention_mask=batch_attention_mask, # [B, L]
            labels=batch_labels,               # [B, L]
        )
            
    def build_generate_inputs_left_align(
    self,
    batch_speech_embedding,
    batch_speech_embedding_length,
):
        """
        左对齐（左 padding）版 generate 输入构建：
        final = [PAD ... PAD | prefix_prompt | speech | postfix_prompt]
        """

        device = batch_speech_embedding.device

        prefix_emb, postfix_emb = self._embed_prompt_parts(device)
        L_prompt = prefix_emb.size(0) + postfix_emb.size(0)

        # ===== collect each sample ===== #
        sample_list = []
        for speech_emb, speech_len in zip(
            batch_speech_embedding,
            batch_speech_embedding_length,
        ):
            speech_emb = speech_emb[:speech_len]    # [T, D]
            L_speech = speech_emb.size(0)

            total_len = L_speech + L_prompt

            sample_list.append(
                dict(
                    speech_emb=speech_emb,
                    prefix_emb=prefix_emb,
                    postfix_emb=postfix_emb,
                    seq_len=total_len
                )
            )

        # ====== compute batch max length ====== #
        batch_size = len(sample_list)
        max_len = max(s["seq_len"] for s in sample_list)
        embed_dim = sample_list[0]["speech_emb"].size(-1)

        # ====== create batch container ===== #
        batch_embeds = torch.zeros(
            (batch_size, max_len, embed_dim),
            dtype=torch.float32,
            device=device,
        )

        batch_attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )

        # ====== LEFT PAD each sample ====== #
        for i, s in enumerate(sample_list):
            L = s["seq_len"]
            pad_len = max_len - L

            # 贴在右侧（左侧填充 pad）
            # positions: [pad ... pad | prefix | speech | postfix]
            batch_embeds[i, pad_len : pad_len + L] = torch.cat(
                [s["prefix_emb"], s["speech_emb"], s["postfix_emb"]], dim=0
            )
            batch_attention_mask[i, pad_len : pad_len + L] = 1

        return dict(
            input_embeds=batch_embeds,            # [B, L, D]
            attention_mask=batch_attention_mask,  # [B, L]
        )
    # '''

    def mask_pad(self,pad_tensor,pad_tensor_len):
        # 假设输入
        # speech_embedding: [B, T, D]
        # speech_embedding_len: [B], 每条语音的有效长度
        B, T, D = pad_tensor.shape

        # print(f"pad_tensor device: {pad_tensor.device}")
        # print(f"pad_tensor_len device: {pad_tensor_len.device}")


        # 1. 创建 mask [B, T], True 表示有效帧
        mask = torch.arange(T, device=pad_tensor.device)[None, :] < pad_tensor_len[:, None]  # [B, T]

        # 2. mask 扩展到 embedding 维度
        mask = mask.unsqueeze(-1)  # [B, T, 1]

        # 3. 将 pad 部分置为 0
        pad_tensor = pad_tensor * mask

        return pad_tensor


    def clean_unpickable_objs(self):
        pass



    
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
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            feats, ilens, phn_labels, phn_lens, y_labels, y_lens = minibatch[:6]
            feats = feats.cuda(args.gpu, non_blocking=True)
            ilens = ilens.cuda(args.gpu, non_blocking=True)

            phn_labels = None
            phn_lens = None


            y_lens = y_lens.cuda(args.gpu, non_blocking=True)

            part_cnt_err, part_cnt_sum = model.module.get_wer(feats, ilens, phn_labels, phn_lens, y_labels, y_lens)
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


def build_beamdecoder(cfg: dict):
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
) -> Union[nn.parallel.DistributedDataParallel, S2P2GAdapterModel, model_zoo.AbsEncoder]:
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

    # 加载tokenizer,包括p_tokenizer和llm_tokenizer
    s2p2g_tokenizer_config = cfg["s2p2g_tokenizer_config"]
    p_tokenizer = load(s2p2g_tokenizer_config["p_tokenizer_path"]) if "p_tokenizer_path" in s2p2g_tokenizer_config else None
    llm_tokenizer = AutoTokenizer.from_pretrained(s2p2g_tokenizer_config["llm_tokenizer_path"], use_fast=False)
    if llm_tokenizer.pad_token_id is None:
        llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    
    # 构建s2p模型

    s2p_netconfigs = cfg.get("s2p_encoder",None)

    if s2p_netconfigs is None:
        s2p_encoder = None
    else:

        s2p_net_kwargs = s2p_netconfigs["kwargs"]  # type:dict

        # The projector configs spell the S2P head size as "n_classes" (the
        # same spelling used by cat.s2p2g.decode_qwen_p2g), while this trainer
        # constructs the encoder through the `num_classes` keyword. Accept both
        # so the published configs run unchanged.
        if "num_classes" in s2p_net_kwargs:
            n_classes = s2p_net_kwargs.pop("num_classes")
        else:
            n_classes = s2p_net_kwargs.pop("n_classes")
        s2p_encoder = getattr(model_zoo, s2p_netconfigs["type"])(
            num_classes = n_classes, **s2p_net_kwargs
        )  # type: model_zoo.AbsEncoder

        if s2p_netconfigs.get("init_model",None) is not None:
            ckpt = s2p_netconfigs.get("init_model",None)
            print(f'ckpt_path: {ckpt}')
            ckpt = torch.load(ckpt)
            ckpt = translate_checkpoint(ckpt['model'],'module.encoder.','',if_add_dot=False)
            s2p_encoder = coreutils.load_checkpoint(s2p_encoder,{"model":ckpt})

    
    # 构建p2g模型
    assert "p2g_model" in cfg
    p2g_enc_configs = cfg["p2g_model"]
    p2g_enc_kwargs = p2g_enc_configs["kwargs"]  # type:dict
    p2g_model = getattr(model_zoo, p2g_enc_configs["type"])(**p2g_enc_kwargs)  # type: model_zoo.AbsEncoder

    if "lora" in p2g_enc_configs:
        lora_cfg = p2g_enc_configs["lora"]
        p2g_model.use_lora = True
        from cat.shared.lora import LinearLayer_LoRA,convert_linear_layer_to_loraX,convert_lora_to_linear_layer,only_optimize_lora_parameters,make_model_gradient_checkpointing_compatible
        if p2g_enc_configs["lora"].get("use_peft",False):
            # PEFT doc:
            # When using LoRA with gradient checkpointing, 
            # you must enable gradient checkpointing before wrapping the model with PEFT.
            p2g_model.llm_model.gradient_checkpointing_enable()
            p2g_model.llm_model = convert_linear_layer_to_loraX(p2g_model.llm_model,**lora_cfg)
        else:
            p2g_model = convert_linear_layer_to_loraX(p2g_model,**lora_cfg)
            # 启用梯度检查点机制
            p2g_model.llm_model = make_model_gradient_checkpointing_compatible(p2g_model.llm_model)
            p2g_model.llm_model.gradient_checkpointing_enable()
    

    # 进行参数冻结
    if s2p_netconfigs is not None and s2p_netconfigs.get("freeze", False):
        s2p_encoder.requires_grad_(False)
    if p2g_enc_configs.get("freeze", False):
        p2g_model.requires_grad_(False)
    # 如果对p2g的embedding层进行了扩充，那么embedding层不应该被冻结
    if p2g_enc_configs.get("train_input_output_embedding",False):
        print("train_input_output_embedding")
        # 输入嵌入
        input_emb = p2g_model.llm_model.get_input_embeddings()  # nn.Embedding
        # 输出嵌入
        output_emb = p2g_model.llm_model.lm_head  # nn.Linear
        # 输入嵌入
        input_emb.weight.requires_grad = True
        # 输出嵌入
        output_emb.weight.requires_grad = True
        if hasattr(output_emb, "bias") and output_emb.bias is not None:
            output_emb.bias.requires_grad = True

    model = S2P2GAdapterModel(
        s2p_encoder,
        p2g_model,
        llm_tokenizer=llm_tokenizer,
        prompt_spec=resolve_prompt_spec(cfg, default_prompt=DEFAULT_PROMPT_TEMPLATE),
        projector_config=cfg["projector_config"],
    )
    
    
    if not dist:
        return model

    assert args is not None, f"You must tell the GPU id to build a DDP model."
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    elif not isinstance(args, dict):
        raise ValueError(f"unsupport type of args: {type(args)}")
    
    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)


    if 'use_fsdp' not in args or not args['use_fsdp']:
        model.cuda(args["gpu"])
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                      device_ids=[args["gpu"]])
        from cat.shared.lora import convert_lora_freeze_weight_to_bf16
        convert_lora_freeze_weight_to_bf16(model)

    model = model.to(torch.bfloat16)

    return model

def get_init_model(s2p_encoder,s2p_netconfigs,p2g_model,p2g_enc_configs,args,use_fsdp=False):
    init_checkpoint = OrderedDict()
    if "init_model" in s2p_netconfigs:
        coreutils.distprint(f"> initialize s2p_encoder from: {s2p_netconfigs['init_model']}", args["gpu"])
        s2p_enc_checkpoint = torch.load(
            s2p_netconfigs["init_model"], 
            map_location=f"cuda:{args['gpu']}" if not use_fsdp else 'cpu'
        )["model"]  # type: OrderedDict
        s2p_enc_checkpoint = translate_checkpoint(s2p_enc_checkpoint, "encoder", "s2p_encoder")
    else:
        s2p_enc_checkpoint = s2p_encoder.state_dict()
        s2p_enc_checkpoint = translate_checkpoint(s2p_enc_checkpoint, "encoder", "module.s2p_encoder.model")
    if use_fsdp:
        s2p_enc_checkpoint = translate_checkpoint(s2p_enc_checkpoint,'module.s2p_encoder','s2p_encoder')
    init_checkpoint.update(s2p_enc_checkpoint)
    del s2p_enc_checkpoint

    if "init_model" in p2g_enc_configs:
        coreutils.distprint(f"> initialize p2g model from: {p2g_enc_configs['init_model']}", args["gpu"])
        p2g_checkpoint = torch.load(
            p2g_enc_configs["init_model"], 
            map_location=f"cuda:{args['gpu']}" if not use_fsdp else 'cpu'
        )["model"]  # type: OrderedDict
        p2g_checkpoint = translate_checkpoint(p2g_checkpoint, "model", "p2g_model")
    else:
        p2g_checkpoint = translate_checkpoint(p2g_model.state_dict(), "model", "module.p2g_model.llm_model")
    if use_fsdp:
        p2g_checkpoint = translate_checkpoint(p2g_checkpoint,'module.p2g_model','p2g_model')
    init_checkpoint.update(p2g_checkpoint)
    del p2g_checkpoint
    return init_checkpoint


# FIXME: following codes will be removed soon or later
########## COMPATIBLE ###########
# fmt: off
def translate_checkpoint(state_dict: OrderedDict, old_string: str, new_string: str,if_add_dot=True) -> OrderedDict:
    """Translate checkpoint of previous version of RNN-T so that it could be loaded with the new one."""
    if if_add_dot:
        old_string = old_string + '.'
        new_string = new_string + '.'
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # print(f'old_string: {old_string},k: {k}')
        if old_string in k:
            k = k.replace(old_string, new_string, 1)
            new_state_dict[k] = v
    return new_state_dict
# fmt: on
#################################

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
