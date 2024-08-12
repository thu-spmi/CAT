#!/bin/bash

# 设置相关路径和文件名
audio_dir="/home/dlk/code/asr/cat_base_asr_copy/MightLJSPeech/MightLJSpeech-1.1/wavs"
text_file="/home/dlk/code/asr/cat_base_asr_copy/MightLJSPeech/MightLJSpeech-1.1"
output_dir="./data/src"
datasets=("train" "dev" "test")



for dataset_name in "${datasets[@]}";do
    # 创建输出目录
    mkdir -p $output_dir/$dataset_name

    # 生成 wav.scp 文件
    while IFS=$'\t' read -r filename text; do
        # 获取完整的音频文件路径
        audio_path="$audio_dir/$filename.wav"
        # 检查音频文件是否存在
        if [ ! -f "$audio_path" ]; then
            echo "Error: $audio_path does not exist."
            continue
        fi
        # 将文件名和音频路径写入 wav.scp
        echo -e "$filename\t$audio_path" >> "$output_dir/$dataset_name/wav.scp"

        # 将音频文件名和对应的文本写入 text 文件，两列用\t分隔
        echo -e "$filename\t$text" >> "$output_dir/$dataset_name/text"
    done < "$text_file/${dataset_name}_data.txt"

    # 生成 text 文件
    #cut -f 2 "$text_file/${dataset_name}_data.txt" > "$output_dir/$dataset_name/text"

    echo "wav.scp and text files generated successfully in $output_dir."
done

data_dir=./data/src

for ti in dev train test;do
  awk '{print $1,$1}' $data_dir/${ti}/text > $data_dir/${ti}/utt2spk
  cp $data_dir/${ti}/utt2spk $data_dir/${ti}/spk2utt

  mv $data_dir/$ti/wav.scp $data_dir/$ti/wav_mp3.scp
  awk '{print $1 "\tffmpeg -i " $2 " -f wav -ar 16000 -ab 16 -ac 1 - |"}' $data_dir/$ti/wav_mp3.scp > $data_dir/$ti/wav.scp

done

bash utils/data/data_prep_kaldi.sh \
    data/src/{train,dev,test} \
    --feat-dir=data/fbank \
    --nj=16 \
    --not-apply-cmvn \
    