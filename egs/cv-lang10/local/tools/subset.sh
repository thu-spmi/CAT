#!bin/bash
# The script select any hours of subset from training data
# set -x -u
language_dir=$1  # data path that contains `text`, `wav.scp` and `utt2dur`
sub_dur=$2  # data hours
mode=0  # mode={0,1} `0` means selecting any hours ;`1` means excluding ang hours
input_data_dir=$language_dir/excluded_train  # excluded_train
utt2dur_file=$language_dir/excluded_train/utt2dur    # utt2dur 


  total_desired_duration=$sub_dur*3600  # $sub_dur hours

#   total_desired_duration=$sub_dur*60  # $sub_dur minutes

#   total_desired_duration=$sub_dur  # $sub_dur seconds

output_data_dir=$language_dir/excluded_train_sub_${sub_dur}m   # sub_excluded_train
mkdir $output_data_dir

awk '{print $1}' "$utt2dur_file" | shuf > $output_data_dir/shuffled_utt_ids.txt
awk '{print $1}' "$utt2dur_file" | sort > $output_data_dir/sorted_utt_ids.txt

current_total_duration=0
while read -r utt_id; do
    utt_duration=$(grep -P "^${utt_id} " "$utt2dur_file" | awk '{print $2}')
    current_total_duration=$(echo "$current_total_duration + $utt_duration" | bc -l)
    if (( $(echo "$current_total_duration >= $total_desired_duration" | bc -l) )); then
        break
    fi
    echo "$utt_id"
done < $output_data_dir/shuffled_utt_ids.txt > $output_data_dir/selected_utt_ids.txt
rm $output_data_dir/shuffled_utt_ids.txt

sort $output_data_dir/selected_utt_ids.txt -o $output_data_dir/selected_utt_ids_new.txt
comm -3 $output_data_dir/selected_utt_ids_new.txt $output_data_dir/sorted_utt_ids.txt > $output_data_dir/unselected_utt_ids.txt
rm $output_data_dir/sorted_utt_ids.txt

cd /opt/kaldi/egs/wsj/s5
if [ ${mode} -le 0 ];then
    utils/subset_data_dir.sh --utt-list $output_data_dir/selected_utt_ids.txt $input_data_dir $output_data_dir
else
    utils/subset_data_dir.sh --utt-list $output_data_dir/unselected_utt_ids.txt $input_data_dir $output_data_dir
fi