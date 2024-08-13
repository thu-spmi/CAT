# train tokenizer and pickle data
# python utils/pipeline/asr.py exp/Whistle-sub-FT --sta 1 --sto 2

# paper pretrain model
# you could download pt_tokenizer and pt_model from 
# https://github.com/thu-spmi/CAT/blob/master/egs/cv-lang10/exp/Multilingual/Multi._phoneme_S/readme.md

# we should modify pt model classfier layer and then we need to modified the hyper-p.json["train"]["option"]["init_model"] = ft_model_path
# python local/process_model_for_subword_ft.py --pt_model_path --output_model_path --vocab_size


# train model
# python utils/pipeline/asr.py exp/Whistle-sub-FT --sta 3 --sto 3

# decode w/o lm
# python utils/pipeline/asr.py exp/Whistle-sub-FT --sta 4 --sto 4
                                                                                                                                               

# decode with lm
# cal_logit
# python utils/pipeline/asr.py exp/Whistle-sub-FT --sta 4 --sto 4
# to decode
# bash exp/bpe_wfst_run.sh --exp_dir exp2/Whistle-sub-FT --lm_dir exp/decode_lm --word_list dict/word_list-2 --dataset_name test-2
