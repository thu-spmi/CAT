# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Lukuan Dong

"""CTC decode module

NOTE (Huahuan): currently, bs=1 is hard-coded.

Reference:
https://github.com/parlance/ctcdecode
"""





from ..shared import coreutils

from ..shared.data import S2P2GTestDataset,S2P2GTestCollate



import os
import time
import pickle

import kaldiio
import argparse
from collections import OrderedDict
from tqdm import tqdm
from typing import *


import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader



import os

from transformers import Qwen3ForCausalLM


# 保证打印时不省略内容
torch.set_printoptions(
    precision=15,       # 小数位数
    sci_mode=False,    # 禁止科学计数法
    linewidth=200,     # 每行最大字符数（防止换行）
    threshold=float('inf')  # 打印全部元素
)


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()


    if args.gpu:
        world_size = torch.cuda.device_count()
        if args.nj != -1 and args.nj < world_size:
            world_size = args.nj
    else:
        if args.nj == -1:
            world_size = max(os.cpu_count() // 2, 1)
        else:
            world_size = args.nj
    assert world_size > 0
    args.world_size = world_size

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

    if args.gpu:
        model = None
    else:
        model = build_model(args)
        model.share_memory()

    mp.spawn(worker, nprocs=world_size, args=(args, q_data, q_out, model))

    producer.join()
    consumer.join()
    del q_data
    del q_out


def dataserver(args, q: mp.Queue):

    testset = S2P2GTestDataset(args.input_scp)
    n_frames = sum(testset.get_seq_len())

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=(args.world_size if args.gpu else args.world_size // 8),
        collate_fn=S2P2GTestCollate()
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
    # print (n_frames)


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
    
def process_output(hypo):
    try:
        # print(hypo)
        process_res =  hypo.split(' | ')
        # process_res =  hypo.split('en:')

        if len(process_res) > 1:
            return(process_res[1].strip())
        else:
            process_res =  hypo.split('|')
            # process_res =  hypo.split(':')
            if len(process_res) > 1:
                return(process_res[1].strip())
            else:
                print(hypo)
                return ""
    except:
        print(hypo)
        return ""

def worker(
    pid: int,
    args: argparse.Namespace,
    q_data: mp.Queue,
    q_out: mp.Queue,
    model,
):
    torch.set_num_threads(args.thread_per_woker)
    if args.gpu:
        device = pid
        torch.cuda.set_device(device)
        model = build_model(args).cuda(device)
    else:
        assert model is not None
        device = "cpu"

    results = {}
    s2p_sp_time = 0
    p2g_sp_time = 0
    
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    # {'uid': {0: (-10.0, 'a b c'), 1: (-12.5, 'a b c d')}}
    with torch.no_grad():
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
        

        while True:
            batch = q_data.get(block=True)
            if batch is None:
                break
            key, x, x_len = batch

            batch_size = x.size(0)
            
            x = x.to(device=device)
            x_len = x_len.to(device=device)

            # x = x.to(device=device,dtype=torch.float16)
            # x_len = x_len.to(device=device,dtype=torch.float16)

            _,speech_embedding_len,speech_embedding = model.s2p_model(x,x_len,return_last_hidden_state=True)

            speech_embedding_len = speech_embedding_len.to(torch.int32)

            speech_embedding = model.mask_pad(speech_embedding,speech_embedding_len)

            speech_align_embedding,speech_align_len = model.projector(speech_embedding,speech_embedding_len)

            generate_inputs_left_align = model.build_generate_inputs_left_align(speech_align_embedding,speech_align_len)
            
            model_out = model.p2g_model.generate(inputs_embeds=generate_inputs_left_align["input_embeds"].to(torch.bfloat16),attention_mask=generate_inputs_left_align["attention_mask"].to(torch.bfloat16),generation_config=generation_config)

            scores = model_out["sequences_scores"] # batch_size * beam_size

            text_hypos = model.llm_tokenizer.batch_decode(model_out["sequences"], skip_special_tokens=True)

            beam_size = args.beam_size
                
            q_out.put(
                tuple(
                    (
                        key[i],
                        {
                            bid: (
                                score.tolist(),
                                hypo
                            )
                            # for bid, (score, hypo) in enumerate(zip(scores, text_hypos))
                            for bid, (score, hypo) in enumerate(zip(scores[i*beam_size:(i+1)*beam_size], text_hypos[i*beam_size:(i+1)*beam_size]))
                        }
                    )
                    for i in range(batch_size)
                ),
                block=True,
            )

            del batch
            del model_out
            

    q_out.put(None, block=True)
    print(f"S2P time: {s2p_sp_time:.4f} seconds")
    print(f"P2G time: {p2g_sp_time:.4f} seconds")
    if args.store_ark:
        output_dir = os.path.join(os.path.dirname(args.output_prefix),"ark")
        os.makedirs(output_dir, exist_ok=True)
        kaldiio.save_ark(os.path.join(output_dir, f"decode.{pid+1}.ark"), results)




@torch.no_grad()
def build_model(args: argparse.Namespace):
    assert (
        args.resume is not None
    ), "Trying to decode with uninitialized parameters. Add --resume"
    import importlib

    # print(args.built_model_by)

    interface = importlib.import_module(args.built_model_by)

    # print("use build_model func")

    model = interface.build_model(coreutils.readjson(args.config), dist=False)

    if args.init_model is not None:
        
        checkpoint = torch.load(
            args.init_model, map_location="cpu"
        )  # type: OrderedDict

        init_model_ckpt = OrderedDict()
        for k, v in checkpoint["model"].items():
            if "module." in k:
                k = k.replace("module.", "", 1)
                init_model_ckpt[k] = v
            else:
                init_model_ckpt[k] = v
        
        
        print(f"try to load init from {args.init_model}")
        model.load_state_dict(init_model_ckpt,strict=False)
        print(f"finish load init from {args.init_model}")
        


    # for debug 注释掉
    checkpoint = torch.load(args.resume, map_location="cpu") 
    model = coreutils.load_checkpoint(model, checkpoint,strict=False) #为了加载lora模块，设置strict=False，此处需要注意

    model.p2g_model = model.p2g_model.to(torch.bfloat16)


    # model = model.to(torch.float16)

    model.eval()
    return model


def _parser():
    parser = coreutils.basic_trainer_parser(
        prog="Seq2Seq decoder.", training=False, isddp=False
    )

    parser.add_argument("--input_scp", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="./decode")
    parser.add_argument("--lm-path", type=str, default=None, help="Path to KenLM model.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="The 'alpha' value for LM integration, a.k.a. the LM weight",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="The 'beta' value for LM integration, a.k.a. the penalty of tokens.",
    )
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
    parser.add_argument("--n_fold", type=int, default=8)
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--thread-per-woker", type=int, default=1)
    parser.add_argument(
        "--built-model-by",
        type=str,
        default="cat.s2p2g.train_s2p2g_adapter",
        help="Tell where to import build_model() function. defautl: cat.s2p2g.train_s2p2g_adapter",
    )
    parser.add_argument("--use_tkm", action="store_true", default=False)
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--store_ark", type=bool, default=False, help="whether store logits as ark file.")
    return parser


if __name__ == "__main__":
    main()
