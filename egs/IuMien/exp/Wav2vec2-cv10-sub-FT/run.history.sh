# Firstly,you could download pretrain model from https://github.com/thu-spmi/CAT/tree/master/egs/cv-lang10/exp/Multilingual/Wav2vec-lang10
# and then we should modify pt model classfier layer 
# python local/process_model_for_subword_ft.py --pt_model_path --output_model_path --vocab_size


# train
# python utils/pipeline/asr.py exp/Wav2vec2-cv10-sub-FT --sta 1 --sto 3
# decode w/o lm
# python utils/pipeline/asr.py exp/Wav2vec2-cv10-sub-FT --sta 4 --sto 4

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
# python utils/pipeline/asr.py exp/Wav2vec2-cv10-sub-FT --sta 4 --sto 4
# to decode
# bash exp/bpe_wfst_run.sh --exp_dir exp/Wav2vec2-cv10-sub-FT --lm_dir exp/decode_lm --word_list dict/word_list-2 --dataset_name test-2_raw
