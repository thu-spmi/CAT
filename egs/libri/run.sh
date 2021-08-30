#!/bin/bash

# Copyright 2021 Tsinghua University
# Author: Hongyu Xiang, Huahuan Zheng
# Apache 2.0.
# This script implements CTC-CRF training on LibriSpeech dataset.
# It's updated to v2 by Huahuan Zheng in 2021, based on CAT branch v1 egs/libri/run.sh

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

. ./cmd.sh
. ./path.sh

stage=1
stop_stage=100
data=/path/to/librispeech
lm_src_dir=/path/to/librispeech_lm

NODE=$1
if [ ! $NODE ]; then
    NODE=0
fi

if [ $NODE == 0 ]; then
    if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
        for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
            local/download_and_untar.sh $data $data_url $part || exit 1
        done
        # download the LM resources
        local/download_lm.sh $lm_url data/local/lm || exit 1
    fi

    if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
        # format the data as Kaldi data directories
        for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
            # use underscore-separated names in data directories.
            local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g) || exit 1
        done

        fbankdir=fbank
        for part in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
            steps/make_fbank.sh --cmd "$train_cmd" --nj 40 data/$part exp/make_fbank/$part $fbankdir || exit 1
            steps/compute_cmvn_stats.sh data/$part exp/make_fbank/$part $fbankdir || exit 1
        done

        # ... and then combine the two sets into a 460 hour one
        utils/combine_data.sh \
            data/train_clean_460 data/train_clean_100 data/train_clean_360 || exit 1

        # combine all the data
        utils/combine_data.sh \
            data/train_960 data/train_clean_460 data/train_other_500 || exit 1
    fi

    if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    # copy the LM resources
    local/copy_lm.sh $lm_src_dir data/local/lm || exit 1
    local/prepare_dict_ctc.sh data/local/lm data/local/dict_phn || exit 1
    ctc-crf/ctc_compile_dict_token.sh data/local/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;
    local/format_lms.sh --src-dir data/lang_phn data/local/lm || exit 1

    # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
    echo "3" > data/lang_phn/oov.int
    utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
        data/lang_phn data/lang_phn_tglarge || exit 1
    utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
        data/lang_phn data/lang_phn_fglarge || exit 1

    langdir=data/lang_phn_tgsmall
    fsttablecompose $langdir/L.fst $langdir/G.fst | fstdeterminizestar --use-log=true | \
        fstminimizeencoded | fstarcsort --sort_type=ilabel > $langdir/LG.fst || exit 1;
    fsttablecompose $langdir/T.fst $langdir/LG.fst > $langdir/TLG.fst || exit 1;

    fi

    data_tr=data/train_tr95
    data_cv=data/train_cv05

    if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
        utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train_960  data/train_tr95 data/train_cv05 || exit 1

        python3 ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_tr/text "<UNK>" > $data_tr/text_number || exit 1
        python3 ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_cv/text "<UNK>" > $data_cv/text_number || exit 1
        echo "convert text_number finished"

        # prepare denominator
        cat $data_tr/text_number | sort -k 2 | uniq -f 1 > $data_tr/unique_text_number
        mkdir -p data/den_meta
        chain-est-phone-lm ark:$data_tr/unique_text_number data/den_meta/phone_lm.fst || exit 1
        python3 ctc-crf/ctc_token_fst_corrected.py den data/lang_phn/tokens.txt | fstcompile | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst || exit 1
        fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst || exit 1
        echo "prepare denominator finished"

        path_weight $data_tr/text_number data/den_meta/phone_lm.fst > $data_tr/weight || exit 1
        path_weight $data_cv/text_number data/den_meta/phone_lm.fst > $data_cv/weight || exit 1
        echo "prepare weight finished"
    fi

    if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
        mkdir -p data/all_ark
        for set in test_clean test_other dev_clean dev_other; do
            eval data_$set=data/$set
        done
        for set in test_clean test_other dev_clean dev_other cv tr; do
            tmp_data=`eval echo '$'data_$set`
            feats="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$tmp_data/utt2spk scp:$tmp_data/cmvn.scp scp:$tmp_data/feats.scp ark:- |"
          
            ark_dir=$(readlink -f data/all_ark)/$set.ark
            copy-feats "$feats" "ark,scp:$ark_dir,data/all_ark/$set.scp" || exit 1
        done
    fi

    if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
        mkdir -p data/pickle
        python3 ctc-crf/convert_to.py -f=pickle --describe='L//4' --filer=1750 \
            data/all_ark/cv.scp $data_cv/text_number $data_cv/weight data/pickle/cv.pickle || exit 1
        python3 ctc-crf/convert_to.py -f=pickle --describe='L//4' --filer=1750 \
            data/all_ark/tr.scp $data_tr/text_number $data_tr/weight data/pickle/tr.pickle || exit 1
    fi

fi


PARENTDIR='.'
dir="exp/libri_phone"
DATAPATH=$PARENTDIR/data/

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    unset CUDA_VISIBLE_DEVICES

    if [[ $NODE == 0 && ! -f $dir/scripts.tar.gz ]]; then
        echo ""
        tar -zcf $dir/scripts.tar.gz $(readlink ctc-crf) $0
    elif [ $NODE == 0 ]; then
        echo ""
        echo "'$dir/scripts.tar.gz' already exists."
        echo "If you want to update it, please manually rm it then re-run this script."
    fi

    # uncomment the following line if you want to use specified GPUs
    # CUDA_VISIBLE_DEVICES="0"                      \
    python3 ctc-crf/train.py --seed=0               \
        --world-size 1 --rank $NODE                 \
        --batch_size=128                            \
        --dir=$dir                                  \
        --config=$dir/config.json                   \
        --data=$DATAPATH                            \
        || exit 1
fi

if [ $NODE -ne 0 ]; then
    exit 0
fi


nj=$(nproc)
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    for set in test_clean test_other dev_clean dev_other; do
        mkdir -p $dir/logits/$set
        ark_dir=$(readlink -f $dir/logits/$set)
        python3 ctc-crf/calculate_logits.py                 \
            --resume=$dir/ckpt/bestckpt.pt                  \
            --config=$dir/config.json                       \
            --nj=$nj --input_scp=data/all_ark/$set.scp      \
            --output_dir=$ark_dir                           \
            || exit 1

        mkdir -p $dir/decode_${set}_tgsmall
        ln -snf $ark_dir $dir/decode_${set}_tgsmall/logits
        ctc-crf/decode.sh  --stage 1 \
            --cmd "$decode_cmd" --nj $nj --acwt 1.0 \
            data/lang_phn_tgsmall data/$set data/all_ark/$set.scp $dir/decode_${set}_tgsmall || exit 1

        lm=tgmed
        rescore_dir=$dir/decode_${set}_$lm
        mkdir -p $rescore_dir
        local/lmrescore.sh --cmd "$train_cmd" data/lang_phn_{tgsmall,$lm} data/$set $dir/decode_${set}_{tgsmall,$lm} $nj || exit 1;

        for lm in tglarge fglarge; do
            rescore_dir=$dir/decode_${set}_$lm
            mkdir -p $rescore_dir
            local/lmrescore_const_arpa.sh --cmd "$train_cmd" data/lang_phn_{tgsmall,$lm} data/$set $dir/decode_${set}_{tgsmall,$lm} $nj || exit 1;
        done
    done

    for set in test_clean test_other dev_clean dev_other; do
        grep WER $dir/decode_${set}_*/wer_* | utils/best_wer.sh
    done
fi
