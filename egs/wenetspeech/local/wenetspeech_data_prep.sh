#!/usr/bin/env bash
# Copyright 2021  Xiaomi Corporation (Author: Yongqing Wang)
#                 Seasalt AI, Inc (Author: Guoguo Chen)
#                 Mobvoi Inc(Author: Di Wu, Binbin Zhang)
#                 ASLP, NWPU (Author: Hang Lyu)

set -e -u
set -o pipefail

train_subset=M
test_subsets="DEV TEST_NET TEST_MEETING"

filter_by_id() {
  idlist=$1
  input=$2
  output=$3
  field=1
  if [ $# -eq 4 ]; then
    field=$4
  fi
  cat $input | perl -se '
    open(F, "<$idlist") || die "Could not open id-list file $idlist";
    while(<F>) {
      @A = split;
      @A>=1 || die "Invalid id-list file line $_";
      $seen{$A[0]} = 1;
    }
    while(<>) {
      @A = split;
      @A > 0 || die "Invalid file line $_";
      @A >= $field || die "Invalid file line $_";
      if ($seen{$A[$field-1]}) {
        print $_;
      }
    }' -- -idlist="$idlist" -field="$field" >$output ||
    (echo "$0: filter_by_id() error: $input" && exit 1) || exit 1
}

subset_data_dir() {
  utt_list=$1
  src_dir=$2
  dest_dir=$3
  mkdir -p $dest_dir || exit 1
  # wav.scp text segments utt2dur utt2spk spk2utt
  filter_by_id $utt_list $src_dir/utt2dur $dest_dir/utt2dur ||
    (echo "$0: subset_data_dir() error: $src_dir/utt2dur" && exit 1) || exit 1
  filter_by_id $utt_list $src_dir/text $dest_dir/text ||
    (echo "$0: subset_data_dir() error: $src_dir/text" && exit 1) || exit 1
  filter_by_id $utt_list $src_dir/segments $dest_dir/segments ||
    (echo "$0: subset_data_dir() error: $src_dir/segments" && exit 1) || exit 1
  awk '{print $2}' $dest_dir/segments | sort | uniq >$dest_dir/reco
  filter_by_id $dest_dir/reco $src_dir/wav.scp $dest_dir/wav.scp ||
    (echo "$0: subset_data_dir() error: $src_dir/wav.scp" && exit 1) || exit 1
  rm -f $dest_dir/reco
  paste <(cat $dest_dir/text | cut -f1) <(cat $dest_dir/text | cut -f1) \
    >$dest_dir/utt2spk || exit 1
  utils/utt2spk_to_spk2utt.pl $dest_dir/utt2spk >$dest_dir/spk2utt || exit 1
}

wenetspeech_dir=$1
data_dir=$2
corpus_dir=$3

declare -A subsets
subsets=(
  [L]="train_l"
  [M]="train_m"
  [S]="train_s"
  [W]="train_w"
  [DEV]="dev"
  [TEST_NET]="test_net"
  [TEST_MEETING]="test_meeting")

echo "$0: Split data to train, dev, test_net, and test_meeting"
[ ! -f $corpus_dir/utt2subsets ] &&
  echo "$0: No such file $corpus_dir/utt2subsets!" && exit 1

for label in $train_subset $test_subsets; do
  if [ ! ${subsets[$label]+set} ]; then
    echo "$0: Subset $label is not defined in WenetSpeech.json." && exit 1
  fi
  subset=${subsets[$label]}
  [ ! -d $data_dir/$subset ] && mkdir -p $data_dir/$subset
  cat $corpus_dir/utt2subsets |
    awk -v s=$label '{for (i=2;i<=NF;i++) if($i==s) print $0;}' \
      >$corpus_dir/${subset}_utt_list || exit 1
  subset_data_dir $corpus_dir/${subset}_utt_list \
    $corpus_dir $data_dir/$subset || exit 1
  utils/fix_data_dir.sh $data_dir/$subset || exit 1
done

echo "$0: Done"
exit 0
