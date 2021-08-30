#!/bin/bash

# Copyright 2021 Tsinghua University
# Author: Chengrui Zhu, Huahuan Zheng
# Apache 2.0.
# This script implements CTC-CRF training on Mozilla Commonvoice dataset.

. ./cmd.sh
. ./path.sh

stage=1
stop_stage=100

# experiment home
DIR=`pwd`
# number of parallel jobs
nj=20

data_de=/path/to/commonvoice/de

g2p_lexicon=local/lexicon.txt
lexicon=local/lexicon_ge.txt

mkdir -p data/
awk '{print $1}' $lexicon > data/pure_word
pure_word=$DIR/data/pure_word

lang=de
train_set=train
dev_set=dev
test_set=test
recog_set="${dev_set} ${test_set}"

NODE=$1
if [ ! $NODE ]; then
    NODE=0
fi

if [ $NODE == 0 ]; then
    if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
        echo "Data Preparation and FST Construction"
        python3 local/resample.py --prev_tr $data_de/validated.tsv --prev_dev $data_de/dev.tsv \
            --to_tr $data_de/resampled_tr.tsv --to_dev $data_de/resampled_dev.tsv

        for part in "test" "resampled_dev" "resampled_tr"; do
            # use underscore-separated names in data directories.
            local/data_prep.pl $data_de ${part} data/${part} || exit 1;

            # rm punctuations
            cat data/${part}/text | sed 's/"//g' | \
                sed 's/,//g' | sed 's/\.//g' | sed 's/\?//g' | sed 's/\!//g' | sed 's/…//g' | \
                sed 's/;//g' | sed 's/  / /g' | sed 's/  / /g' | sed 's/ $//g' | \
                sed "s/’/'/g" > data/${part}/text_fil || exit 1;
            mv data/${part}/text_fil data/${part}/text || exit 1;
        done

        utils/copy_data_dir.sh data/resampled_tr data/${train_set} || exit 1;
        utils/copy_data_dir.sh data/resampled_dev data/${dev_set} || exit 1;
        utils/filter_scp.pl --exclude data/${dev_set}/wav.scp data/${train_set}/wav.scp > data/${train_set}/temp_wav.scp || exit 1;
        utils/filter_scp.pl --exclude data/${test_set}/wav.scp data/${train_set}/temp_wav.scp > data/${train_set}/wav.scp || exit 1;
        utils/fix_data_dir.sh data/${train_set} || exit 1;

        local/mozilla_prepare_phn_dict.sh $g2p_lexicon || exit 1;
        ctc-crf/ctc_compile_dict_token.sh --dict-type "phn" \
            data/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;
        local/mozilla_train_lms.sh data/train_de/text data/dict_phn/lexicon.txt data/local/local_lm || exit 1;

        local/mozilla_format_local_lms.sh --lang-suffix "phn" || exit 1;
        local/mozilla_decode_graph.sh data/local/local_lm data/lang_phn data/lang_phn_test || exit 1;
    fi 

    if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
        echo "Fbank Feature Generation"
        # Perturb the speaking speed to achieve data augmentation
        echo "Generate 3 way speed perturbation."
        utils/perturb_data_dir_speed.sh 0.9 data/${train_set} data/temp1 || exit 1;
        utils/perturb_data_dir_speed.sh 1.0 data/${train_set} data/temp2 || exit 1;
        utils/perturb_data_dir_speed.sh 1.1 data/${train_set} data/temp3 || exit 1;
        utils/combine_data.sh data/${train_set}_sp data/temp1 data/temp2 data/temp3 || exit 1;
        rm -r data/temp1 data/temp2 data/temp3
        if ! utils/validate_data_dir.sh --no-feats --no-text data/${train_set}_sp; then
            echo "$0: Validation failed.  If it is a sorting issue, try the option '--always-include-prefix true'."
            exit 1
        fi

        fbankdir=fbank
        for set in ${test_set} ${dev_set} ${train_set}_sp; do
            steps/make_fbank.sh --cmd "$train_cmd" --nj $nj data/$set exp/make_fbank/$set $fbankdir || exit 1;
            utils/fix_data_dir.sh data/$set || exit;
            steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
        done
    fi

    train_set=${train_set}_sp

    data_tr=data/$train_set
    data_cv=data/$dev_set

    if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
        # Convert word sequences to label sequences according to lexicon_numbers.txt and text files in data/lang_phn
        # ...the result will be placed in $data_tr/ and $data_cv/
        python3 ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_tr/text "<UNK>" > $data_tr/text_number
        python3 ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_cv/text "<UNK>" > $data_cv/text_number
        echo "Convert text_number finished"

        # Prepare denominator
        # sort the text_number file, and then remove the duplicate lines
        cat $data_tr/text_number | sort -k 2 | uniq -f 1 > $data_tr/unique_text_number
        mkdir -p data/den_meta
        # generate phone_lm.fst, a phone-based language model
        chain-est-phone-lm ark:$data_tr/unique_text_number data/den_meta/phone_lm.fst
        # generate the correct T.fst, called T_den.fst
        python3 ctc-crf/ctc_token_fst_corrected.py den data/lang_phn/tokens.txt | fstcompile | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst
        # compose T_den.fst and phone_lm.fst into den_lm.fst
        fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst
        echo "Prepare denominator finished"

        # calculate and save the weight for each label sequence based on text_number and phone_lm.fst
        path_weight $data_tr/text_number data/den_meta/phone_lm.fst > $data_tr/weight
        path_weight $data_cv/text_number data/den_meta/phone_lm.fst > $data_cv/weight
        echo "Prepare weight finished"

    fi

    if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
        mkdir -p data/all_ark
        data_test=data/$test_set
        for set in test cv tr; do
            tmp_data=`eval echo '$'data_$set`
            feats="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$tmp_data/utt2spk scp:$tmp_data/cmvn.scp scp:$tmp_data/feats.scp ark:- |"
            ark_dir=$(readlink -f data/all_ark)/$set.ark
            copy-feats "$feats" "ark,scp:$ark_dir,data/all_ark/$set.scp" || exit 1
        done
        echo "Copy feats finished"
    fi

    if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
        mkdir -p data/pickle
        python3 ctc-crf/convert_to.py -f=pickle --describe='L//4' --filer=1500 \
            data/all_ark/cv.scp $data_cv/text_number $data_cv/weight data/pickle/cv.pickle || exit 1
        python3 ctc-crf/convert_to.py -f=pickle --describe='L//4' --filer=1500 \
            data/all_ark/tr.scp $data_tr/text_number $data_tr/weight data/pickle/tr.pickle || exit 1
    fi
fi

PARENTDIR='.'
dir="exp/cv_de_phone"
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
        --batch_size=256                            \
        --dir=$dir                                  \
        --config=$dir/config.json                   \
        --data=$DATAPATH                            \
        || exit 1
fi

if [ $NODE -ne 0 ]; then
    exit 0
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    for set in test; do
        ark_dir=$dir/logits/${set}
        mkdir -p $ark_dir
        ark_dir=$(readlink -f $ark_dir)
        python3 ctc-crf/calculate_logits.py                 \
            --resume=$dir/ckpt/bestckpt.pt.pt               \
            --config=$dir/config.json                       \
            --nj=$nj --input_scp=data/all_ark/${set}.scp    \
            --output_dir=$ark_dir                           \
            || exit 1
        echo "Logits generated."

        mkdir -p $dir/decode_${set}
        ln -snf $ark_dir $dir/decode_${set}/logits
        ctc-crf/decode.sh  --stage 1 \
            --cmd "$decode_cmd" --nj $nj --acwt 1.0 \
            data/lang_phn_test data/$set data/all_ark/$set.ark $dir/decode_${set} || exit 1

        lm=bd_tg
        mkdir -p $dir/decode_${set}_$lm
        steps/lmrescore.sh --cmd "$decode_cmd" --mode 3 data/lang_phn_test{,_$lm} data/${set} $dir/decode_${set}{,_$lm} || exit 1;

        lm=bd_fgconst
        mkdir -p $dir/decode_${set}_$lm
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_phn_test{,_$lm} data/${set} $dir/decode_${set}{,_$lm} || exit 1;

        grep WER $dir/decode_${set}*/wer_* | utils/best_wer.sh
    done
fi
