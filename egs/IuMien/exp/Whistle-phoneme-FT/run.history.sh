# train tokenizer and pickle data
# python utils/pipeline/asr.py exp2/Whistle-phoneme-FT --sta 1 --sto 2

# paper pretrain model
# you could download pt_tokenizer and pt_model from 
# https://github.com/thu-spmi/CAT/blob/master/egs/cv-lang10/exp/Multilingual/Multi._phoneme_S/readme.md

# pt_tokenizer_path=
# ft_tokenizer_path=
# pt_model_path=
# output_model_path=
# python exp/Whistle-phoneme-FT/unpack_mulingual_param.py $pt_tokenizer_path $ft_tokenizer_path $pt_model_path $output_model_path  --mode flatphone



# train model
# python utils/pipeline/asr.py exp2/Whistle-phoneme-FT --sta 3 --sto 3

# per test
# python utils/pipeline/asr.py exp2/Whistle-phoneme-FT --sta 4 --sto 4

# use wfst decode
# First, you need to modify the hyper-p.json file.
# "infer": {
#             "bin": "cat.ctc.cal_logit",
#             "option": {
#                 "beam_size": 32,
#                 "nj": 16,
#                 "store_ark": true
#             }
#         },
# python utils/pipeline/asr.py exp/Whistle-phoneme-FT --sta 4 --sto 4
# bash exp/lexicon_wfst_run.sh --exp_dir exp/Whistle-phoneme-FT --lm_dir exp/decode_lm  --dataset_name test
