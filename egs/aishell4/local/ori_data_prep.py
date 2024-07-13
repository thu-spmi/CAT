# Copyright 2020 Tsinghua SPMI Lab / Tasi
# Apache 2.0.
# Author: Xiangzhu Kong (kongxiangzhu99@gmail.com)
#
# Description:
#   This script processes AISHELL-4 dataset by segmenting audio files based on TextGrid annotations.
#   It extracts non-overlapping speech segments, saves them as separate WAV files, and generates corresponding 
#   Kaldi format files including wav.scp, text, utt2spk, and spk2utt. The script also handles directory creation 
#   and logging of the processing steps.

import textgrid as tg
from pydub import AudioSegment
import soundfile as sf
import os
import re
import logging


# 设置日志级别和日志格式
logging.basicConfig(level=logging.INFO, format='%(message)s')

from tqdm import tqdm
from intervaltree import IntervalTree

def maximize_nonoverlapping_count(intervals):
    """
    Maximize the count of non-overlapping intervals.

    Args:
        intervals (list): List of tuples representing intervals (start, end).

    Returns:
        list: List of non-overlapping intervals.
    """
    # 创建 IntervalTree 对象并插入所有间隔
    tree = IntervalTree()
    for start, end in intervals:
        tree[start:end] = None

    # 初始化结果列表
    non_overlapping_intervals = []

    # 遍历每个间隔，检查是否与其他间隔有交集
    for start, end in intervals:
        if len(tree[start:end]) == 1:
            non_overlapping_intervals.append((start, end))

    return non_overlapping_intervals


def main(aishell4_path):
    id=1
    punctuation = '&!！,;:?？"\'、，.。；'
    

    if not os.path.exists('./data/src'):
        os.mkdir('./data/src')
    if not os.path.exists('./data/src/train'):
        os.mkdir('./data/src/train')
    if not os.path.exists('./data/src/dev'):
        os.mkdir('./data/src/dev')
    if not os.path.exists('./data/src/test'):
        os.mkdir('./data/src/test')

    #创建分段后的波形存储目录
    if not os.path.exists(aishell4_path +'/seg-wav'):
        os.mkdir(aishell4_path +'/seg-wav')
    if not os.path.exists(aishell4_path +'/seg-wav/train'):
        os.mkdir(aishell4_path +'/seg-wav/train')
    if not os.path.exists(aishell4_path +'/seg-wav/dev'):
        os.mkdir(aishell4_path +'/seg-wav/dev')
    if not os.path.exists(aishell4_path +'/seg-wav/test'):
        os.mkdir(aishell4_path +'/seg-wav/test')   

    tr_wav_scp = open("./data/src/train/wav.scp", 'w+')
    tr_text_scp = open('./data/src/train/text', 'w+')
    tr_utt2spk = open('./data/src/train/utt2spk', 'w+')
    tr_spk2utt = open('./data/src/train/spk2utt', 'w+')

    cv_wav_scp = open("./data/src/dev/wav.scp", 'w+')
    cv_text_scp = open('./data/src/dev/text', 'w+')
    cv_utt2spk = open('./data/src/dev/utt2spk', 'w+')
    cv_spk2utt = open('./data/src/dev/spk2utt', 'w+')

    test_wav_scp = open("./data/src/test/wav.scp", 'w+')
    test_text_scp = open('./data/src/test/text', 'w+')
    test_utt2spk = open('./data/src/test/utt2spk', 'w+')
    test_spk2utt = open('./data/src/test/spk2utt', 'w+')

    dataset='train'
    for root,dirs,files in os.walk(aishell4_path +"/"+dataset+"/TextGrid"):
        for file in tqdm(files, desc="Processing files in " + dataset):
            if file not in ["20200622_M_R002S07C01.TextGrid", "20200710_M_R002S06C01.TextGrid"]:#此文件暂时有问题
                if file.endswith(".TextGrid"):  # 检查文件扩展名是否为".TextGrid"
                    extracted_text = re.search(r'^(.*?)\.', file).group(1)
                    text_file = os.path.join(root,file)
                    logging.info("%s", text_file)
                    wav_file=aishell4_path +"/"+dataset+"/wav/"+text_file.split("/")[-1][:-9]+'.flac'
                    tgrid = tg.TextGrid.fromFile(text_file)
                    sound = AudioSegment.from_file(wav_file)
                    #sound, samplerate = sf.read(wav_file)

                    intervals = []
                    d = dict()

                    for item in tgrid:
                        for utt in item:
                            if len(utt.mark.strip())>0:
                                intervals.append((utt.minTime,utt.maxTime))
                                d[(utt.minTime,utt.maxTime)]=utt.mark
                    non_overlap_intervals = maximize_nonoverlapping_count(intervals)
                    non_overlap_intervals = set(non_overlap_intervals)
                    logging.info("\noverlap ratio: %f", 1 - len(non_overlap_intervals) / len(intervals))

                    for item in tgrid:
                        spk = item.name
                        for utt in item:
                            if len(utt.mark.strip())>0 and (utt.minTime,utt.maxTime) in non_overlap_intervals:

                                if id%20!=0:
                                    wav_name =aishell4_path +'/seg-wav/train/'+str(id)+".wav"
                                    start_time = float(utt.minTime)*1000
                                    end_time = float(utt.maxTime)*1000
                                    wav = sound[start_time:end_time]
                                    wav.export(wav_name, format="wav")
                                    
                                    tr_wav_scp.writelines(extracted_text +'-'+ spk+'-'+str(id) +"\t"+ wav_name +"\n")
                                    tr_utt2spk.writelines(spk+'-'+str(id) +"\t"+ spk + "\n")
                                    tr_spk2utt.writelines(spk +"\t"+ spk+'-'+str(id) + "\n")
                                    tr_text_scp.writelines(extracted_text +'-'+ spk+'-'+str(id)+"\t"+re.sub(r'[{}]+'.format(punctuation),'',utt.mark.rstrip()) + "\n")
                                else:
                                    wav_name =aishell4_path +'/seg-wav/dev/'+str(id)+".wav"
                                    start_time = float(utt.minTime)*1000
                                    end_time = float(utt.maxTime)*1000
                                    wav = sound[start_time:end_time]
                                    wav.export(wav_name, format="wav")
                                    
                                    cv_wav_scp.writelines(extracted_text +'-'+ spk+'-'+str(id) +"\t"+ wav_name +"\n")
                                    cv_utt2spk.writelines(spk+'-'+str(id) +"\t"+ spk + "\n")
                                    cv_spk2utt.writelines(spk +"\t"+ spk+'-'+str(id) + "\n")
                                    cv_text_scp.writelines(extracted_text +'-'+ spk+'-'+str(id)+"\t"+re.sub(r'[{}]+'.format(punctuation),'',utt.mark.rstrip()) + "\n")
                                                    
                                id += 1

    dataset="test"
    for root,dirs,files in os.walk(aishell4_path +"/"+dataset+"/TextGrid"):
        for file in tqdm(files, desc="Processing files in " + dataset):
            if file.endswith(".TextGrid"):
                extracted_text = re.search(r'^(.*?)\.', file).group(1)
                text_file = os.path.join(root,file)
                wav_file=aishell4_path +"/"+dataset+"/wav/"+text_file.split("/")[-1][:-9]+'.flac'
                tgrid = tg.TextGrid.fromFile(text_file)
                sound = AudioSegment.from_file(wav_file)

                intervals = []
                d = dict()

                for item in tgrid:
                    for utt in item:
                        if len(utt.mark.strip())>0:
                            intervals.append((utt.minTime,utt.maxTime))
                            d[(utt.minTime,utt.maxTime)]=utt.mark
                non_overlap_intervals = maximize_nonoverlapping_count(intervals)
                non_overlap_intervals = set(non_overlap_intervals)
                logging.info("\noverlap ratio: %f", 1 - len(non_overlap_intervals) / len(intervals))

                for item in tgrid:
                    spk = item.name
                    for utt in item:
                        if len(utt.mark.strip())>0 and (utt.minTime,utt.maxTime) in non_overlap_intervals:
                            wav_name =aishell4_path +'/seg-wav/'+dataset+'/'+str(id)+".wav"
                            start_time = float(utt.minTime)*1000
                            end_time = float(utt.maxTime)*1000
                            wav = sound[start_time:end_time]
                            wav.export(wav_name, format="wav")
                            
                            test_wav_scp.writelines(extracted_text +'-'+ spk+'-'+str(id) +"\t"+ wav_name +"\n")
                            test_utt2spk.writelines(spk+'-'+str(id) +"\t"+ spk + "\n")
                            test_spk2utt.writelines(spk +"\t"+ spk+'-'+str(id) + "\n")
                            test_text_scp.writelines(extracted_text +'-'+ spk+'-'+str(id)+"\t"+re.sub(r'[{}]+'.format(punctuation),'',utt.mark.rstrip()) + "\n")
                        
                            id += 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_aishell4>")
        sys.exit(1)

    aishell4_path = sys.argv[1]
    main(aishell4_path)