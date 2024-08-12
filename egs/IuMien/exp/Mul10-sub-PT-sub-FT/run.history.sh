# Firstly,you could download pretrain model from https://github.com/thu-spmi/CAT/blob/master/egs/cv-lang10/exp/Multilingual/Multi._subword/readme.md
# and then we should modify pt model classfier layer and then you need to modified the hyper-p.json["train"]["option"]["init_model"] = ft_model_path
# python local/process_model_for_subword_ft.py --pt_model_path --output_model_path --vocab_size

# train
# python utils/pipeline/asr.py exp/Mul10-sub-PT-sub-FT --sta 1 --sto 3

# decode w/o lm
# python utils/pipeline/asr.py exp/Mul10-sub-PT-sub-FT --sta 4 --sto 4


# decode with lm
# First, you need to modify the hyper-p.json file.
# "infer": {
#             "bin": "cat.ctc.cal_logit",
#             "option": {
#                 "beam_size": 32,
#                 "nj": 16,
#                 "store_ark": true
#             }
#         },
# cal_logit
# python utils/pipeline/asr.py exp/Mul10-sub-PT-sub-FT --sta 4 --sto 4
# to decode
# bash exp/bpe_wfst_run.sh --exp_dir exp/Mul10-sub-PT-sub-FT --lm_dir exp/decode_lm --word_list dict/word_list --dataset_name test

