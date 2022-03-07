#!/bin/bash

# Copyright 2021 Tsinghua University
# Author: Chengrui Zhu, Huahuan Zheng
# Apache 2.0.
# This script implements CTC-CRF training on Mozilla Commonvoice dataset.
set -e
set -o pipefail

. ./cmd.sh
. ./path.sh

stage=1
stop_stage=100
nj=$(nproc)
data="/path/to/commonvoice/de"
lang=de
train_set=train_$(echo $lang | tr - _)
dev_set=dev_$(echo $lang | tr - _)
test_set=test_$(echo $lang | tr - _)
recog_set="$dev_set $test_set"
nbpe=150
bpemode=unigram # `char` for char-based system.


NODE=$1
if [ ! $NODE ]; then
    NODE=0
fi

if [ $NODE == 0 ]; then
    if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Data Preparation and FST Construction"
    python3 local/resample.py --prev_tr $data/validated.tsv --prev_dev $data/dev.tsv \
        --to_tr $data/resampled_tr.tsv --to_dev $data/resampled_dev.tsv || exit 1;

    # Use the same data preparation script from Kaldi
    for part in "test" "resampled_dev" "resampled_tr"; do
        # use underscore-separated names in data directories.
        local/data_prep.pl "${data}" ${part} data/"$(echo "${part}_${lang}" | tr - _)" || exit 1;
    done

    # remove test&dev data from validated sentences
    utils/copy_data_dir.sh data/"$(echo "resampled_tr_${lang}" | tr - _)" data/${train_set} || exit 1;
    utils/copy_data_dir.sh data/"$(echo "resampled_dev_${lang}" | tr - _)" data/${dev_set} || exit 1;
    utils/filter_scp.pl --exclude data/dev_de/wav.scp data/train_de/wav.scp > data/train_de/temp_wav.scp || exit 1;
    utils/filter_scp.pl --exclude data/test_de/wav.scp data/train_de/temp_wav.scp > data/train_de/wav.scp || exit 1;
    utils/fix_data_dir.sh data/train_de || exit 1;

    utils/perturb_data_dir_speed.sh 0.9 data/${train_set} data/temp1 || exit 1;
    utils/perturb_data_dir_speed.sh 0.9 data/${train_set} data/temp1 || exit 1;
    utils/perturb_data_dir_speed.sh 1.0 data/${train_set} data/temp2 || exit 1;
    utils/perturb_data_dir_speed.sh 1.1 data/${train_set} data/temp3 || exit 1;
    utils/combine_data.sh --extra-files utt2uniq data/train_de data/temp1 data/temp2 data/temp3 || exit 1;
    rm -rf data/{temp1,temp2,temp3}

    fi

    if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        echo "Fbank Feature Generation"
        fbankdir=fbank
        for x in dev test train; do
            mkdir -p data/$x
            cp -r data/${x}_de/* data/$x/
            steps/make_fbank.sh --cmd "$train_cmd" --nj $nj data/${x} exp/make_fbank/$x \
                $fbankdir || exit 1;
            utils/fix_data_dir.sh data/${x} || exit 1;
            steps/compute_cmvn_stats.sh data/${x} exp/make_fbank/$x $fbankdir || exit 1
        done  
    fi


    if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then

      echo "stage 3: Dictionary and Json Data Preparation"
      local/mozilla_prepare_bpe_dict.sh || exit 1

      ctc-crf/ctc_compile_dict_token.sh --dict_type "bpe" data/local/dict_bpe \
          data/local/lang_bpe_tmp data/lang_bpe || exit 1;
      echo "Building n-gram LM model."
      
      # train.txt without uttid for training n-gramm
      cat data/train/text_pos | cut -f 2- -d " " - > data/local/dict_bpe/train.txt || exit 1;
      local/mozilla_train_lms.sh data/local/dict_bpe/train.txt data/local/dict_bpe/lexicon.txt data/local/local_lm || exit 1;
      local/mozilla_format_local_lms.sh --lang-suffix "bpe"  || exit 1;
      local/mozilla_decode_graph.sh data/local/local_lm data/lang_bpe data/lang_bpe_test || exit 1;


      for x in train dev; do
          ctc-crf/prep_ctc_trans.py data/lang_bpe/lexicon_numbers.txt data/$x/text_pos \
              "<UNK>" > data/${x}/text_number || exit 1;
      done
      echo "convert text_number finished."

      cat data/train/text_number | sort -k 2 | uniq -f 1 > data/train/unique_text_number || exit 1;
      mkdir -p data/den_meta
      chain-est-phone-lm ark:data/train/unique_text_number data/den_meta/phone_lm.fst || exit 1;
      ctc-crf/ctc_token_fst_corrected.py den data/lang_bpe/tokens.txt | fstcompile \
          | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst || exit 1;
      fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst || exit 1;
      echo "prepare denominator finished"
      path_weight data/train/text_number data/den_meta/phone_lm.fst > data/train/weight || exit 1
      path_weight data/dev/text_number data/den_meta/phone_lm.fst > data/dev/weight || exit 1
      echo "prepare weight finished"

    fi 

    if [ $stage -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        mkdir -p data/all_ark
        for set in test dev train; do
            feats="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:data/${set}/utt2spk scp:data/${set}/cmvn.scp scp:data/${set}/feats.scp ark:- |"
            ark_dir=$(readlink -f data/all_ark)/$set.ark
            copy-feats "$feats" "ark,scp:$ark_dir,data/all_ark/$set.scp" || exit 1
        done
        echo "Copy feats finished"
    fi

    if [ $stage -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        mkdir -p data/pickle
        for set in dev train; do
            python3 ctc-crf/convert_to.py -f=pickle --describe='L//4' --filer=1500 \
                data/all_ark/${set}.scp data/${set}/text_number data/${set}/weight data/pickle/${set}.pickle || exit 1
        done
        # To match the same path
        mv data/pickle/train.pickle data/pickle/tr.pickle || exit 1
        mv data/pickle/dev.pickle data/pickle/cv.pickle || exit 1
    fi
fi


PARENTDIR='.'
dir="exp/cv_de_wp" # `exp/cv_de_char` for char-based system
DATAPATH=$PARENTDIR/data/

if [ $stage -le 6 ] && [ ${stop_stage} -ge 6 ]; then
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
    # CUDA_VISIBLE_DEVICES="0"                        \
    python3 ctc-crf/train.py --seed=0               \
        --world-size 1 --rank $NODE                 \
        --batch_size=256                            \
        --dir=$dir                                  \
        --config=$dir/config.json                   \
        --data=$DATAPATH                            \
        || exit 1
fi

if [ $NODE -ne 0 ]; then
    exit 0
fi


if [ $stage -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    for set in test; do
        ark_dir=$dir/logits/${set}
        mkdir -p $ark_dir
        ark_dir=$(readlink -f $ark_dir)
        python3 ctc-crf/calculate_logits.py                 \
            --resume=$dir/ckpt/bestckpt.pt                  \
            --config=$dir/config.json                       \
            --nj=$nj --input_scp=data/all_ark/${set}.scp    \
            --output_dir=$ark_dir                           \
            || exit 1
        echo "Logits generated."


        lm=tgpr
        mkdir -p $dir/decode_${set}_bd_$lm
        ln -snf $ark_dir $dir/decode_${set}_bd_$lm/logits
        ctc-crf/decode.sh  --stage 1 \
            --cmd "$decode_cmd" --nj $nj --acwt 1.0 \
            data/lang_bpe_${set}_bd_$lm data/$set data/all_ark/$set.ark $dir/decode_${set}_bd_$lm || exit 1

        echo "Decode done."

        lm=fgconst
        mkdir -p $dir/decode_${set}_bd_$lm
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_bpe_${set}_bd_{tgpr,$lm} data/${set} $dir/decode_${set}_bd_{tgpr,$lm} || exit 1;

        echo "4-gram Rescoring done."

        grep WER $dir/decode_${set}*/wer_* | utils/best_wer.sh
    done
fi

