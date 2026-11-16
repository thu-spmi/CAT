# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Lukuan Dong

"""CTC decode module

NOTE (Huahuan): currently, bs=1 is hard-coded.

Reference:
https://github.com/parlance/ctcdecode
"""


import torch.distributed
from ..shared.tokenizer import LexiconTokenizer, PretrainedTokenizer, load
from ..shared import coreutils
from ..shared.encoder import AbsEncoder
from ..shared import encoder as model_zoo
from ..shared.data import sortedScpPadCollate, ScpDataset



import os
import re
import time
import pickle
import kaldiio
import argparse
from collections import OrderedDict
from tqdm import tqdm
from typing import *
from ctcdecode import CTCBeamDecoder

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.nn.functional as F

def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()


    if args.gpu:
        args.world_size = torch.cuda.device_count()
        assert args.world_size  > 0, "No GPU available for decoding."
    else:
        raise NotImplementedError("CPU decoding is not supported yet.")

    try:
        mp.set_start_method("spawn")
    except RuntimeError as re:
        print(re)

    q_data = mp.Queue(maxsize=1)
    producer = mp.Process(target=dataserver, args=(args, q_data))
    producer.start()

    q_out = mp.Queue(maxsize=1)
    consumer = mp.Process(target=datawriter, args=(args, q_out))
    consumer.start()

    mp.spawn(worker, nprocs=args.world_size, args=(args, q_data, q_out))

    producer.join()
    consumer.join()
    del q_data
    del q_out


def dataserver(args, q: mp.Queue):

    testset = ScpDataset(args.input_scp)
    n_frames = sum(testset.get_seq_len())
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=(args.world_size if args.gpu else args.world_size // 8),
        collate_fn=sortedScpPadCollate()
    )

    t_beg = time.time()
    for batch in tqdm(
        testloader, desc="AED decode", total=len(testloader), leave=False
    ):
        for k in batch:
            if isinstance(k, torch.Tensor):
                k.share_memory_()
        q.put(batch, block=True)

    for _ in range(args.world_size + 1):
        q.put(None, block=True)
    t_dur = time.time() - t_beg

    print(
        "Time = {:.2f} s | RTF = {:.2f} ".format(
            t_dur, t_dur * args.world_size / n_frames * 100
        )
    )
    print (n_frames)


def datawriter(args, q: mp.Queue):
    cnt_done = 0
    nbest = {}
    transcript = []

    while True:
        nbestlist = q.get(block=True)  # type: Tuple[str, Dict[int, Tuple[float, str]]]
        if nbestlist is None:
            cnt_done += 1
            if cnt_done == args.world_size:
                break
            continue
        for i in range(len(nbestlist)):
            key, content = nbestlist[i]
            nbest[key] = content
            transcript.append(f"{key}\t{content[0][1]}\n")
        del nbestlist

    with open(args.output_prefix, "w", encoding="utf-8") as fo:
        for l in transcript:
            fo.write(l)

    with open(args.output_prefix + ".nbest", "wb") as fo:
        pickle.dump(nbest, fo)
    

def worker(
    pid: int,
    args: argparse.Namespace,
    q_data: mp.Queue,
    q_out: mp.Queue
):
    torch.set_num_threads(args.thread_per_woker)
    device = pid
    torch.cuda.set_device(device)
    model = build_model(args).cuda(device)

    from transformers import GenerationConfig
    generation_config = GenerationConfig(
                                        _from_model_config=True,
                                        eos_token_id=model.llm_tokenizer.eos_token_id,
                                        pad_token_id=model.llm_tokenizer.pad_token_id,
                                        do_sample=False,
                                        max_new_tokens=512,
                                        length_penalty=1.0,     
                                        num_beams=args.beam_size,
                                        num_return_sequences=args.beam_size,
                                        return_dict_in_generate=True,
                                        output_scores=True,
                                        output_logits=True,
                                        use_cache=True,
                                        repetition_penalty = 1.1,
                                        no_repeat_ngram_size=5,
                                        )
    # non_tonal = True if args.input_scp.split('/')[-3] == 'librispeech' else False
    non_tonal = False
    if non_tonal:
        # create a new tensor with tiny value
        tiny_log_value = torch.log(torch.tensor(torch.finfo(torch.float16).tiny)).to(torch.float32).to(device)
        subClassList = torch.tensor(model.p_tokenizer.get_subClassList, dtype=torch.long, device=device)
        n_base_class = subClassList.size(0)
        vocab_size = model.p_tokenizer.test_vocab_size
    else:
        vocab_size = model.p_tokenizer.vocab_size

    labels = [""] * vocab_size
    phn_decoder = CTCBeamDecoder(
        labels,
        # beam_width=args.s2p_beam_size,
        beam_width=16,
        log_probs_input=True,
        num_processes=2,
    )
    results = {}
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    with torch.no_grad():
        while True:
            batch = q_data.get(block=True)
            if batch is None:
                break
            key, x, x_len = batch
            x = x.to(device)
            
            logits_s2p_enc, logits_lens_s2p_enc = model.s2p_model(x, x_len)
            logits_s2p_enc = torch.log_softmax(logits_s2p_enc, dim=-1)
            logits_lens_s2p_enc = logits_lens_s2p_enc.to(torch.int32)

            if non_tonal:
                # For non-tonal classes, we need to sum the log prob of the sub-classes
                extended_logits = logits_s2p_enc.new_full(logits_s2p_enc.shape[:-1] + (n_base_class,), tiny_log_value)
                extended_logits[0] = torch.cat(
                    [
                        logits_s2p_enc[0].index_select(dim=-1, index=sub).logsumexp(dim=-1, keepdim=True)
                        for sub in subClassList
                    ],
                    dim=-1,
                )
                
                # Set the logits corresponding to sub class to a very small value
                logits_s2p_enc[0].index_fill_(dim=-1,index=subClassList.flatten(),value=tiny_log_value)
                logits_s2p_enc = torch.cat([logits_s2p_enc, extended_logits], dim=-1)

            phn_beam_results, _ , _, phn_result_lens = phn_decoder.decode(
                logits_s2p_enc, logits_lens_s2p_enc
            )

            if args.topk > 0:
                indices = torch.arange(args.topk) 
                phn_beam_results = phn_beam_results[:, indices, :]
                phn_result_lens = phn_result_lens[:, indices]
                
            N, K, T = phn_beam_results.shape
            phn_beam_results = phn_beam_results.view(N*K,-1)
            phn_result_lens = phn_result_lens.view(N*K) 

            logits_s2p_enc = logits_s2p_enc.repeat_interleave(K,dim=0)
            logits_lens_s2p_enc  = logits_lens_s2p_enc.repeat_interleave(K,dim=0)

            log_s2p_probs = model.ctc_loss(logits_s2p_enc.transpose(0, 1), phn_beam_results, logits_lens_s2p_enc, phn_result_lens).to(torch.float64)
            
            # 在p2g时，每条p可以beam解码得到num_beams条g,因此需要对s2p_loss再进行generation_config.num_beams次repeat_interleave
            log_s2p_probs = log_s2p_probs.repeat_interleave(generation_config.num_beams, dim=0).view(N,-1)
            del logits_s2p_enc, logits_lens_s2p_enc, x, x_len

            # if args.p2g_decode_fold > 1: 
            #     p2g_decode_fold = args.p2g_decode_fold
            #     fold_phn_beam_results_list = torch.split(phn_beam_results,p2g_decode_fold,dim=0)
            #     fold_phn_result_lens_list = torch.split(phn_result_lens,p2g_decode_fold,dim=0)

            #     text_hyp = []
            #     text_y_ids_list = []
            #     p2g_beamdecode_logprobs_list = []
            #     max_len = -1
            #     for i in range(len(fold_phn_beam_results_list)):
            #         fold_text_hyp, fold_text_y_ids,fold_p2g_beamdecode_logprobs = model.p2g_decode(fold_phn_beam_results_list[i], fold_phn_result_lens_list[i], generation_config)
            #         fold_p2g_beamdecode_logprobs = torch.sum(fold_p2g_beamdecode_logprobs,-1)

            #         text_hyp.extend(fold_text_hyp)
            #         text_y_ids_list.append(fold_text_y_ids)
            #         p2g_beamdecode_logprobs_list.append(fold_p2g_beamdecode_logprobs)

            #         if max_len < fold_text_y_ids.shape[1]:
            #             max_len = fold_text_y_ids.shape[1]

            #     pad_token = model.llm_tokenizer.pad_token_id
            #     for i in range(len(text_y_ids_list)):
            #         pad_len = max_len - text_y_ids_list[i].shape[1]
            #         if pad_len > 0:
            #             text_y_ids_list[i] = F.pad(text_y_ids_list[i], pad=(0, pad_len), value=pad_token)
            #     batch_beam_g_token_ids = torch.cat(text_y_ids_list,dim=0)
            #     p2g_beamdecode_logprobs = torch.cat(p2g_beamdecode_logprobs_list,dim=0)
            #     log_p2g_probs = p2g_beamdecode_logprobs.view(N,-1)
            # else:
            #     text_hyp, batch_beam_g_token_ids,p2g_beamdecode_logprobs = model.p2g_decode(phn_beam_results, phn_result_lens, generation_config)
            #     log_p2g_probs = torch.sum(p2g_beamdecode_logprobs,-1).view(N,-1)


            try:
                text_hyp, batch_beam_g_token_ids,p2g_beamdecode_logprobs = model.p2g_decode(phn_beam_results, phn_result_lens, generation_config)
                log_p2g_probs = torch.sum(p2g_beamdecode_logprobs,-1).view(N,-1)
            except:
                try:
                    p2g_decode_fold = 8
                    fold_phn_beam_results_list = torch.split(phn_beam_results,p2g_decode_fold,dim=0)
                    fold_phn_result_lens_list = torch.split(phn_result_lens,p2g_decode_fold,dim=0)

                    text_hyp = []
                    text_y_ids_list = []
                    p2g_beamdecode_logprobs_list = []
                    max_len = -1
                    for i in range(len(fold_phn_beam_results_list)):

                        fold_text_hyp, fold_text_y_ids,fold_p2g_beamdecode_logprobs = model.p2g_decode(fold_phn_beam_results_list[i], fold_phn_result_lens_list[i], generation_config)
                        fold_p2g_beamdecode_logprobs = torch.sum(fold_p2g_beamdecode_logprobs,-1)

                        text_hyp.extend(fold_text_hyp)
                        text_y_ids_list.append(fold_text_y_ids)
                        p2g_beamdecode_logprobs_list.append(fold_p2g_beamdecode_logprobs)

                        if max_len < fold_text_y_ids.shape[1]:
                            max_len = fold_text_y_ids.shape[1]

                    pad_token = model.llm_tokenizer.pad_token_id
                    for i in range(len(text_y_ids_list)):
                        pad_len = max_len - text_y_ids_list[i].shape[1]
                        if pad_len > 0:
                            text_y_ids_list[i] = F.pad(text_y_ids_list[i], pad=(0, pad_len), value=pad_token)
                    batch_beam_g_token_ids = torch.cat(text_y_ids_list,dim=0)
                    p2g_beamdecode_logprobs = torch.cat(p2g_beamdecode_logprobs_list,dim=0)
                    log_p2g_probs = p2g_beamdecode_logprobs.view(N,-1)
                except:
                    print(phn_beam_results)
                    print(phn_result_lens)
            

            # print(text_hyp)

            # s2p_loss shape:(batch_size,s2p_topk*p2g_numbeams)
            s2p2g_probs = torch.exp(log_p2g_probs - log_s2p_probs)
            batch_beam_g_token_ids = batch_beam_g_token_ids.view(N, K*generation_config.num_beams, -1)
            
            # 接下来求每个uniq_g的tkm score评分
            # 记录去重后的g_token_ids以及对于的tkm score,内容为[(tkm_score,g_token_ids),]
            batch_uniq_score_g_ids = []
            for batch_index,beam_g_token_ids in enumerate(batch_beam_g_token_ids):
                # 首先需要对p2g的beam generate结果进行去重
                beam_uniq_g, beam_uniq_g_index, beam_uniq_g_counts = torch.unique(beam_g_token_ids, dim=0, sorted=False, return_inverse=True, return_counts=True)
                
                # 每个uniq_g的初始得分为0
                beam_uniq_score_g_token_ids = [[0.,uniq_g] for uniq_g in beam_uniq_g]

                # s2p2g_probs中记录了每个s2p2g序列的概率得分
                # beam_uniq_g_index 中记录了每条p可以经过beamdecode得到哪条uniq_g
                # 遍历beam_uniq_g_index,将s2p2g得分加到对应的uniq_g_tkm得分上
                for g_index,uniq_g_index in enumerate(beam_uniq_g_index):
                    beam_uniq_score_g_token_ids[uniq_g_index][0] += s2p2g_probs[batch_index][g_index].item()

                # 降序排列
                beam_uniq_score_g_token_ids.sort(key=lambda x:x[0],reverse=True)
                batch_uniq_score_g_ids.append(beam_uniq_score_g_token_ids)
            

            # for batch_index, beam_g_token_ids in enumerate(batch_beam_g_token_ids):
            #     # 1. 去重：获取唯一序列、索引和计数
            #     beam_uniq_g, beam_uniq_g_inverse, beam_uniq_g_counts = torch.unique(
            #         beam_g_token_ids, 
            #         dim=0, 
            #         sorted=False, 
            #         return_inverse=True, 
            #         return_counts=True
            #     )
                
            #     # 2. 向量化得分聚合（替代Python循环）
            #     # 获取当前batch的得分
            #     current_probs = s2p2g_probs[batch_index]  # [beam_size]
                
            #     # 使用scatter_add进行高效聚合
            #     # beam_uniq_g_inverse告诉我们每个原始序列对应哪个唯一序列
            #     uniq_scores = torch.zeros(len(beam_uniq_g), dtype=current_probs.dtype, device=current_probs.device)
            #     uniq_scores = uniq_scores.scatter_add_(0, beam_uniq_g_inverse, current_probs)
                
            #     # 3. 排序：使用torch.sort（GPU优化）
            #     sorted_scores, sort_indices = torch.sort(uniq_scores, descending=True)
                
            #     # 4. 构建结果：使用张量索引替代Python循环
            #     sorted_uniq_g = beam_uniq_g[sort_indices]
                
            #     # 5. 转换为所需的列表格式
            #     beam_uniq_score_g_token_ids = [
            #         [score.item(), token_ids] 
            #         for score, token_ids in zip(sorted_scores, sorted_uniq_g)
            #     ]
                
            #     batch_uniq_score_g_ids.append(beam_uniq_score_g_token_ids)


            q_out.put(
                tuple(
                    (
                        key[i],
                        {
                            bid: (
                                score, model.llm_tokenizer.decode(hypo_token_ids, skip_special_tokens=True)
                            )
                            for bid, (score, hypo_token_ids) in enumerate(batch_uniq_score_g_ids[i])
                        }
                    )
                    for i in range(N)  # N: batch size
                ),
                block=True,
            )

    q_out.put(None, block=True)
    if args.store_ark:
        output_dir = os.path.join(os.path.dirname(args.output_prefix),"ark")
        os.makedirs(output_dir, exist_ok=True)
        kaldiio.save_ark(os.path.join(output_dir, f"decode.{pid+1}.ark"), results)

class SPGModel(torch.nn.Module):
    def __init__(
        self,
        p_prompt: str,
        s2p_model: model_zoo.AbsEncoder,
        p2g_model: model_zoo.AbsEncoder,
        llm_tokenizer: PretrainedTokenizer,
        p_tokenizer: LexiconTokenizer
    ):  
        super().__init__()
        self.s2p_model = s2p_model
        self.p2g_model = p2g_model
        self.p_prompt = p_prompt
        self.p_tokenizer = p_tokenizer
        self.llm_tokenizer = llm_tokenizer
        
        if self.llm_tokenizer.pad_token_id is None:
            self.llm_tokenizer.tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id

        self.p2g_model.model.config.eos_token_id = llm_tokenizer.eos_token_id
        self.p2g_model.model.config.pad_token_id = llm_tokenizer.pad_token_id
        self.ctc_loss = torch.nn.CTCLoss(reduction='none',zero_infinity=True)
        # 新的音素映射
        phoneme_list = [
            "a˥", "a˥˩", "a˧˥", "a˧˩˧",
            "e˥", "e˥˩", "e˧˥", "e˧˩˧",
            "i˥", "i˥˩", "i˧˥", "i˧˩˧",
            "o˥", "o˥˩", "o˧˥", "o˧˩˧",
            "u˥", "u˥˩", "u˧˥", "u˧˩˧",
            "y˥", "y˥˩", "y˧˥", "y˧˩˧",
            "ə˥", "ə˥˩", "ə˧˥", "ə˧˩˧",
            "ɛ˥", "ɛ˥˩", "ɛ˧˥", "ɛ˧˩˧",
            "ɤ˥", "ɤ˥˩", "ɤ˧˥", "ɤ˧˩˧",
            "ɹ̩˥", "ɹ̩˥˩", "ɹ̩˧˥", "ɹ̩˧˩˧",
            "ɻ̩˥", "ɻ̩˥˩", "ɻ̩˧˥", "ɻ̩˧˩˧",
            "ʊ˥", "ʊ˥˩", "ʊ˧˥", "ʊ˧˩˧"
        ]
        phoneme_list_replace = [
            "ā", "à", "á", "ǎ",
            "ē", "è", "é", "ě",
            "ī", "ì", "í", "ǐ",
            "ō", "ò", "ó", "ǒ",
            "ū", "ù", "ú", "ǔ",
            "ȳ", "ỳ", "ý", "y̌",
            "ə̄", "ə̀", "ə́", "ə̌",
            "ɛ̄", "ɛ̀", "ɛ́", "ɛ̌",
            "ɤ̄", "ɤ̀", "ɤ́", "ɤ̌",
            "ɹ̩̄", "ɹ̩̀", "ɹ̩́", "ɹ̩̌",
            "ɻ̩̄", "ɻ̩̀", "ɻ̩́", "ɻ̩̌",
            "ʊ̄", "ʊ̀", "ʊ́", "ʊ̌"
        ]
        self.replace_dict = dict(zip(phoneme_list, phoneme_list_replace))
        sorted_phonemes = sorted(self.replace_dict.keys(), key=lambda x: -len(x))
        self.pattern = re.compile(
            r'(\s|^)' + r'(' + '|'.join(re.escape(p) for p in sorted_phonemes) + r')' + r'(?=\s|$)'
        )

    def p2g_decode(self, input_x, lens_input_x, config):
        # p_list = [
        #     self.p_prompt.format(p= ''.join(
        #         self.replace_tones(self.p_tokenizer.decode(
        #             input_x[i][:lens_input_x[i]].tolist())).split()))
        #            for i in range(input_x.size(0))
        # ]

        p_list = [
            self.p_prompt.format(p= self.p_tokenizer.decode(
                    input_x[i][:lens_input_x[i]].tolist()))
                   for i in range(input_x.size(0))
        ]



        self.llm_tokenizer.padding_side = 'left'
        input_p2g = self.llm_tokenizer(p_list, padding='longest',max_length=512,truncation=True,return_tensors="pt")
        p2g_dec_out = self.p2g_model.model.generate(input_p2g.input_ids.cuda(), 
                                                    config,
                                                    attention_mask=input_p2g.attention_mask.cuda()
                                                    ) 
        self.llm_tokenizer.padding_side = 'right'
        max_input_len = input_p2g.input_ids.size(1)
        out_sequences = p2g_dec_out.sequences[:,max_input_len:]
        text_hypos = self.llm_tokenizer.batch_decode(out_sequences, skip_special_tokens=True)
        beam_score = self.p2g_model.model.compute_transition_scores(
                                                out_sequences,
                                                p2g_dec_out.logits,
                                                p2g_dec_out.beam_indices,
                                                normalize_logits=True)
        return text_hypos, out_sequences, beam_score

    def replace_tones(self, phn_seq: str) -> str:
        def replace_phoneme(match):
            prefix = match.group(1)
            phoneme = match.group(2)
            return prefix + self.replace_dict.get(phoneme, phoneme)
        
        return self.pattern.sub(replace_phoneme, phn_seq)

@torch.no_grad()
def build_model(args: argparse.Namespace):
    assert (
        args.resume is not None
    ), "Trying to decode with uninitialized parameters. Add --resume"

    cfg = coreutils.readjson(args.config)

    # load tokenizer
    p_tokenizer = load(cfg["decoder"]["s2p_tokenizer"])
    llm_tokenizer = load(args.tokenizer)
    # tonal_language = cfg["decoder"].get(["tonal_language"], False)
    
    # build s2p model
    s2p_netconfigs = cfg["decoder"]["s2p_model"]
    s2p_net_kwargs = s2p_netconfigs["kwargs"]
    s2p_model = getattr(model_zoo,s2p_netconfigs["type"])(num_classes = s2p_net_kwargs.pop("n_classes"), **s2p_net_kwargs)
    s2p_model.load_state_dict(
        translate_checkpoint(
            torch.load(s2p_netconfigs["init_model"], weights_only=False)['model'],
            'module.encoder.',
            ''
        ))

    # build p2g model
    p2g_enc_configs = cfg["trainer"]["model"]
    p2g_model = getattr(model_zoo, p2g_enc_configs["type"])(**p2g_enc_configs["kwargs"] )

    print(f'p2g resume: {args.resume}')

    p2g_model.load_state_dict(
        translate_checkpoint(
            torch.load(args.resume, weights_only=False)['model'],
            'module.model.',
            ''
        ))

    # for old SSKM codebase compatibility
    # p2g_model.load_state_dict(
    #     translate_checkpoint(
    #         torch.load(args.resume, weights_only=False)['model'],
    #         'module.p2g_model.',
    #         ''
    #     ))

    # build s2p beam decoder
    # beamDecoder_cfg = cfg["decoder"]["s2p_decoder"]
    # beamDecoder_cfg["num_classes"] = beamDecoder_cfg.pop("n_classes")
    # phn_decoder = build_beamdecoder(beamDecoder_cfg)

    prompt = coreutils.readjson(os.path.join(os.path.dirname(args.config), "hyper-p.json"))["train"]["option"]["prompt"]

    # p2g_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) 
    # p2g_model = coreutils.load_checkpoint(p2g_model, p2g_checkpoint,strict=True) #为了加载lora模块，设置strict=False，此处需要注意
    # p2g_model = model.p2g_model.to(torch.bfloat16)

    # if "lora" in p2g_enc_configs["kwargs"] and p2g_enc_configs["kwargs"]["lora"]:
    #     from ..shared.lora import convert_lora_to_linear_layer
    #     p2g_model = convert_lora_to_linear_layer(p2g_model,True)
    
    model = SPGModel(s2p_model=s2p_model,
                      p2g_model=p2g_model,
                      p_tokenizer=p_tokenizer,
                      llm_tokenizer=llm_tokenizer,
                      p_prompt=prompt
                      )

    print('try to move to fp16')
    model.p2g_model = model.p2g_model.to(torch.float16)

    model.eval()

    model.p2g_model.fuse_lora()

    return model

def translate_checkpoint(state_dict: OrderedDict, old_string: str, new_string: str) -> OrderedDict:
    """Translate checkpoint of previous version of RNN-T so that it could be loaded with the new one."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if old_string in k:
            k = k.replace(old_string, new_string, 1)
            new_state_dict[k] = v
    return new_state_dict

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
    labels = [""] * cfg["num_classes"]

    return CTCBeamDecoder(
        labels=labels,
        beam_width=cfg.get("beam_size", 16),
        num_processes=cfg.get("num_processes", 6),
        log_probs_input=cfg.get("log_probs_input", True)
    )

def _parser():
    parser = coreutils.basic_trainer_parser(
        prog="Seq2Seq decoder.", training=False, isddp=False
    )

    parser.add_argument("--input_scp", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="./decode")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--p2g-decode-fold", type=int, default=1)
    parser.add_argument(
        "--do-normalize",
        action="store_true",
        default=False,
        help="Do the log-softmax normalization before beam search.",
    )
    parser.add_argument(
        "--load_ckpt_from_s2p_p2g",
        action="store_true",
        default=False,
        help="Do the log-softmax normalization before beam search.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer model file. See cat/shared/tokenizer.py for details.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU to do inference. Default: False.",
    )
    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--p2g_decode_fold", type=int, default=0)
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--thread-per-woker", type=int, default=1)
    parser.add_argument(
        "--built-model-by",
        type=str,
        default="cat.s2p2g.train_qwen_p2g",
        help="Tell where to import build_model() function. defautl: cat.s2p2g.train_qwen_p2g",
    )
    parser.add_argument("--use_tkm", action="store_true", default=False)
    parser.add_argument("--store_ark", type=bool, default=False, help="whether store logits as ark file.")
    return parser


if __name__ == "__main__":
    main()
