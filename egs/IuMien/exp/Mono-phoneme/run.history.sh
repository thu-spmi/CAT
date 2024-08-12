# train model
# python utils/pipeline/asr.py exp/Mono-phoneme --sta 1 --sto 3

# test model per
# python utils/pipeline/asr.py exp/Mono-phoneme --sta 4 --sto 4


# test model wer
# First, you need to modify the hyper-p.json file.
# "infer": {
#             "bin": "cat.ctc.cal_logit",
#             "option": {
#                 "beam_size": 32,
#                 "nj": 16,
#                 "store_ark": true
#             }
#         },
# python utils/pipeline/asr.py exp/Mono-phoneme --sta 4 --sto 4
# bash exp/lexicon_wfst_run.sh --exp_dir exp/Mono-phoneme --lm_dir exp/decode_lm  --dataset_name test
