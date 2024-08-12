# train model
# python utils/pipeline/asr.py exp2/Mono-subword --sta 1 --sto 3
# decode w/o lm
# python utils/pipeline/asr.py exp2/Mono-subword --sta 4 --sto 4


# decode with lm
# cal_logit
# First, you need to modify the hyper-p.json file.
# "infer": {
#             "bin": "cat.ctc.cal_logit",
#             "option": {
#                 "beam_size": 32,
#                 "nj": 16,
#                 "store_ark": true
#             }
#         },
# python utils/pipeline/asr.py exp/Mono-subword --sta 4 --sto 4
# to decode
# bash exp/bpe_wfst_run.sh --exp_dir exp/Mono-subword --lm_dir exp/decode_lm --word_list dict/word_list --dataset_name test

