#!/bin/bash

# Copyright 2021 Tsinghua University
# Author: Chengrui Zhu
# Apache 2.0.
# This script implements multi/cross-lingual training.
set -e

. ./cmd.sh
. ./path.sh

stage=1
stop_stage=100
H=$(pwd) #exp home
nj=48    #parallel jobs#!/usr/bin/env bash

NODE=$1
if [ ! $NODE ]; then
    NODE=0
fi

lang=(de it fr es)
datadir=/path/to/cv-corpus-5.1-2020-06-22/

# You should first download the lexicons and unpack to your directory:
# https://drive.google.com/file/d/1o_xHo6ntlaB8sTJ_QOkwZDUglIcfPNuF/view?usp=sharing
saved_dict="saved_dict"
dict_tmp=data/local/dict_tmp

[ ! -d $saved_dict ] &&
    echo "no such lexicon directory: $saved_dict" &&
    echo "download the lexicons first: https://drive.google.com/file/d/1o_xHo6ntlaB8sTJ_QOkwZDUglIcfPNuF/view?usp=sharing" &&
    exit 1

mkdir -p $dict_tmp

if [ $NODE == 0 ]; then
    if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
        echo "Data Preparation and FST Construction"
        # Use the same datap prepatation script from Kaldi, create directory data/train, data/dev and data/test,
        for x in ${lang[@]}; do
            train_set=train_"$(echo "${x}" | tr - _)"
            train_dev=dev_"$(echo "${x}" | tr - _)"
            test_set=test_"$(echo "${x}" | tr - _)"
            for usg in validated test dev; do
                # use underscore-separated names in data directories.
                if [ "$x" = "fr" ]; then
                    local/data_prep.pl "${datadir}/$x" ${usg}_filt data/"$(echo "${usg}_${x}" | tr - _)"
                else
                    local/data_prep.pl "${datadir}/$x" ${usg} data/"$(echo "${usg}_${x}" | tr - _)"
                fi
                cat data/"$(echo "${usg}_${x}" | tr - _)"/text | sed 's/"//g' | sed 's/,//g' | sed 's/\.//g' | sed 's/\?//g' | sed 's/\!//g' | sed 's/…//g' | sed 's/;//g' | sed 's/  / /g' | sed 's/  / /g' | sed 's/ $//g' | sed "s/’/'/g" >data/"$(echo "${usg}_${x}" | tr - _)"/text_fil
                mv data/"$(echo "${usg}_${x}" | tr - _)"/text_fil data/"$(echo "${usg}_${x}" | tr - _)"/text
                if [ "$x" = 'it' ] || [ "$x" = 'fr' ]; then
                    tr A-Z a-z <data/${usg}_${x}/text >data/${usg}_${x}/text.tmp
                    mv data/${usg}_${x}/text.tmp data/${usg}_${x}/text
                fi
            done
            utils/copy_data_dir.sh data/"$(echo "validated_${x}" | tr - _)" data/${train_set}
            utils/filter_scp.pl --exclude data/${train_dev}/wav.scp data/${train_set}/wav.scp >data/${train_set}/temp_wav.scp
            utils/filter_scp.pl --exclude data/${test_set}/wav.scp data/${train_set}/temp_wav.scp >data/${train_set}/wav.scp
            utils/fix_data_dir.sh data/${train_set}

            cat data/${train_set}/text | awk '{$1="";print $0}' | sed 's/ /\n/g' | sort -u >$dict_tmp/wordlist_${x}
        done

        # merge all langs data into one
        for usg in train test dev; do
            mkdir -p data/${usg}
            touch data/${usg}/spk2utt
            touch data/${usg}/text
            touch data/${usg}/utt2spk
            touch data/${usg}/wav.scp
            for x in ${lang[@]}; do
                cat data/${usg}_${x}/spk2utt >>data/${usg}/spk2utt
                cat data/${usg}_${x}/text >>data/${usg}/text
                cat data/${usg}_${x}/utt2spk >>data/${usg}/utt2spk
                cat data/${usg}_${x}/wav.scp >>data/${usg}/wav.scp
            done
            utils/fix_data_dir.sh data/${usg}
        done
    fi

    if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

        mul_lexicon=$saved_dict/lexicon_mul.txt
        local/mozilla_prepare_phn_dict.sh $mul_lexicon || exit 1
        ctc-crf/ctc_compile_dict_token.sh --dict-type "phn" \
            data/dict_phn data/local/lang_phn_tmp data/lang_phn

        for x in ${lang[@]}; do
            lex=$saved_dict/lexicon_${x}.txt
            local/mozilla_prepare_phn_dict.sh $lex data/train_${x}/text data/dict_phn_${x} || exit 1
            ctc-crf/ctc_compile_dict_token.sh --dict-type "phn" \
                data/dict_phn_${x} data/local/lang_phn_${x}_tmp data/lang_phn_${x} || exit 1
            local/mozilla_train_lms.sh data/train_${x}/text data/dict_phn_${x}/lexicon.txt \
                data/local/local_lm_${x} || exit 1
            local/mozilla_format_local_lms.sh --lang-suffix "phn" $x || exit 1
            local/mozilla_decode_graph.sh data/local/local_lm_${x} data/lang_phn_${x} \
                data/lang_phn_${x}_test || exit 1
        done
    fi

    if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
        echo "FBank Feature Generation"
        #perturb the speaking speed to achieve data augmentation
        utils/data/perturb_data_dir_speed_3way.sh data/train data/train_sp
        utils/data/perturb_data_dir_speed_3way.sh data/dev data/dev_sp
        fbankdir=fbank

        for x in train_sp dev_sp test test_de test_fr test_es test_it; do
            steps/make_fbank.sh --cmd "$train_cmd" --nj 40 --fbank-config conf/vggblstm_fbank.conf data/$x exp/make_fbank/$x $fbankdir || exit 1
            utils/fix_data_dir.sh data/$x || exit                                     #filter and sort the data files
            steps/compute_cmvn_stats.sh data/$x exp/make_fbank/$x $fbankdir || exit 1 #achieve cmvn normalization
        done

        for x in ${lang[@]}; do
            for usg in train dev; do
                mkdir -p data/${usg}_${x}_sp
                srcdir=data/${usg}_sp
                for f in $(ls $srcdir); do
                    f=$(basename $f)
                    if [ -f $srcdir/$f ]; then
                        if [ $f = "frame_shift" ] || [ $f = "cmvn.scp" ]; then
                            cp $srcdir/$f data/${usg}_${x}_sp/${f}
                        else
                            grep "_${x}_" $srcdir/$f >data/${usg}_${x}_sp/$f
                        fi
                    fi
                done
                utils/fix_data_dir.sh data/${usg}_${x}_sp
            done
        done
    fi

    data_tr=data/train_sp
    data_cv=data/dev_sp

    if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
        #convert word sequences to label sequences according to lexicon_numbers.txt and text files in data/lang_phn
        #the result will be placed in $data_tr/ and $data_cv/
        for x in "" ${lang[@]}; do
            [ "$x" != "" ] && x=_${x}
            data_train=data/train${x}_sp
            data_dev=data/dev${x}_sp
            python3 ctc-crf/prep_ctc_trans.py data/lang_phn${x}/lexicon_numbers.txt $data_train/text "<UNK>" >$data_train/text_number
            python3 ctc-crf/prep_ctc_trans.py data/lang_phn${x}/lexicon_numbers.txt $data_dev/text "<UNK>" >$data_dev/text_number
        done
        echo "convert text_number finished"

    fi

    if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
        ark_dir="data/all_ark"
        mkdir -p $ark_dir
        mkdir -p $ark_dir/arks
        scp_fmt="$ark_dir/%s_%s.scp"
        for usg in "test" "dev" "train"; do
            mkdir -p $ark_dir/$usg
            for x in ${lang[@]}; do
                f_scp=$(printf $scp_fmt $usg $x)
                [ -f $f_scp ] &&
                    echo "scp file: $f_scp exist with $(wc -l $f_scp | awk '{print $1}') lines, skip re-computing" &&
                    echo "... check the number of lines, if it's incorrect, rm the file and run the script again." &&
                    continue
                dir=data/${usg}
                if [ $usg != "test" ]; then
                    dir=${dir}_${x}_sp
                else
                    dir=${dir}_${x}
                fi

                mkdir -p $ark_dir/$usg/$x
                $cmd JOB=1:1 /dev/null \
                    copy-feats "ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$dir/utt2spk scp:$dir/cmvn.scp scp:$dir/feats.scp ark:- | \
                    add-deltas ark:- ark:- | \
                    subsample-feats --n=3 ark:- ark:- |" \
                    "ark,scp:$ark_dir/arks/${usg}_${x}.ark,$f_scp" || exit 1

                echo "Process done: $f_scp"
            done
            # merge sub-lang scps into one
            [ $usg != "test" ] &&
                (for x in ${lang[@]}; do cat $(printf $scp_fmt $usg $x); done >$ark_dir/${usg}.scp)
        done
        echo "copy-feats finished"
    fi

    if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
        for x in ${lang[@]} ""; do
            [ "$x" != "" ] && x=_${x}
            den_meta_dir=data/den_meta${x}
            data_train=data/train${x}_sp
            data_dev=data/dev${x}_sp
            lang_phn=data/lang_phn${x}
            train_scp=data/all_ark/train${x}.scp
            dev_scp=data/all_ark/dev${x}.scp
            train_pkl=data/pickle/train${x}.pickle
            dev_pkl=data/pickle/dev${x}.pickle
            mkdir -p $den_meta_dir
            [ -f $train_pkl ] && [ -f $dev_pkl ] && continue

            cat $data_train/text_number | sort -k 2 | uniq -f 1 >$data_train/unique_text_number
            chain-est-phone-lm ark:$data_train/unique_text_number $den_meta_dir/phone_lm.fst
            python3 ctc-crf/ctc_token_fst_corrected.py den $lang_phn/tokens.txt | fstcompile | fstarcsort --sort_type=olabel >$den_meta_dir/T_den.fst
            fstcompose $den_meta_dir/T_den.fst $den_meta_dir/phone_lm.fst >$den_meta_dir/den_lm.fst
            echo "prepare denominator finished"
            path_weight $data_train/text_number $den_meta_dir/phone_lm.fst >$data_train/weight
            path_weight $data_dev/text_number $den_meta_dir/phone_lm.fst >$data_dev/weight

            mkdir -p data/pickle
            python3 ctc-crf/convert_to.py -f pickle \
                $train_scp $data_train/text_number $data_train/weight $train_pkl || exit 1
            python3 ctc-crf/convert_to.py -f pickle \
                $dev_scp $data_dev/text_number $data_dev/weight $dev_pkl || exit 1
        done
    fi

fi

PARENTDIR='.'
dir="exp/mc_flatphone/"
DATAPATH=$PARENTDIR/data/

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    if [[ $NODE == 0 && ! -f $dir/scripts.tar.gz ]]; then
        echo ""
        tar -zcf $dir/scripts.tar.gz $(readlink ctc-crf) $0
    elif [ $NODE == 0 ]; then
        echo ""
        echo "'$dir/scripts.tar.gz' already exists."
        echo "If you want to update it, please manually rm it then re-run this script."
    fi

    # uncomment the following line if you want to use specified GPUs
    # CUDA_VISIBLE_DEVICES="0,1"                      \
    python3 ctc-crf/train.py --seed=0 \
        --world-size 1 --rank $NODE \
        --batch_size=128 \
        --dir=$dir \
        --config=$dir/config.json \
        --trset=data/pickle/train.pickle \
        --devset=data/pickle/dev.pickle \
        --data=$DATAPATH ||
        exit 1
fi

finetune_dir="exp/mc_flatphone_finetune/"
if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    # finetune
    if [[ $NODE == 0 && ! -f $dir/scripts.tar.gz ]]; then
        echo ""
        tar -zcf $dir/scripts.tar.gz $(readlink ctc-crf) $0
    elif [ $NODE == 0 ]; then
        echo ""
        echo "'$dir/scripts.tar.gz' already exists."
        echo "If you want to update it, please manually rm it then re-run this script."
    fi

    for x in de; do
        CUDA_VISIBLE_DEVICES=0,1,2 \
            python3 ctc-crf/train.py --seed=0 \
            --world-size 1 --rank $NODE \
            --batch_size=128 \
            --resume=$dir/ckpt/bestckpt.pt \
            --den-lm=data/den_meta_${x}/den_lm.fst \
            --mc-conf=conf/mc_flatphone_finetune_${x}.json \
            --trset=data/pickle/train_${x}.pickle \
            --devset=data/pickle/dev_${x}.pickle \
            --dir=$finetune_dir \
            --config=$dir/config.json \
            --data=data/train_${x}_sp
    done
fi

[ $NODE -ne 0 ] && exit 0

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    for x in de; do
        scp=data/all_ark/test_${x}.scp
        ark_dir=$finetune_dir/decode_${x}_test_bd_tgpr/logits
        mkdir -p $ark_dir
        # CUDA_VISIBLE_DEVICES=0,1,2,3 \
        python3 ctc-crf/calculate_logits.py \
            --mc-conf=conf/mc_flatphone_finetune_eval_${x}.json \
            --resume=$finetune_dir/ckpt/infer.pt \
            --config=$finetune_dir/config.json \
            --nj=$nj --input_scp=$scp \
            --output_dir=$ark_dir ||
            exit 1

        ctc-crf/decode.sh --stage 1 --cmd "$decode_cmd" --nj $nj --acwt 1.0 data/lang_phn_${x}_test_bd_tgpr \
            data/test_${x} data/all_ark/test_${x}.ark $finetune_dir/decode_${x}_test_bd_tgpr || exit 1
    done
fi
