# Copyright 2020 Tsinghua SPMI Lab 
# Apache 2.0.
# Author: Xiangzhu Kong(kongxiangzhu99@gmail.com)
#
# Description:
#   This script filters the wav.scp file based on uids from a text file. 
#   It removes lines in the wav.scp file that correspond to the uids listed in the text file.
#   The key functions include loading uids from the text file, filtering the wav.scp file, and writing the filtered lines to an output file.


import os

def filter_wav_scp(text_file_path, wav_scp_file_path, output_wav_scp_file_path):
    """
    Filters lines in wav.scp file based on uids found in the text file and writes the filtered lines to an output file.

    Args:
        text_file_path (str): Path to the text file containing uids.
        wav_scp_file_path (str): Path to the wav.scp file to be filtered.
        output_wav_scp_file_path (str): Path to the output wav.scp file after filtering.

    """
    # Load uids from text file
    uids = set()
    with open(text_file_path, 'r') as text_file:
        for line in text_file:
            uid, _ = line.strip().split('\t')
            uids.add(uid)

    # Filter wav.scp lines based on uids
    filtered_lines = []
    with open(wav_scp_file_path, 'r') as wav_scp_file:
        for line in wav_scp_file:
            uid, _ = line.strip().split('\t')
            if uid in uids:
                filtered_lines.append(line)

    # Write filtered lines to output file
    with open(output_wav_scp_file_path, 'w') as output_wav_scp_file:
        output_wav_scp_file.writelines(filtered_lines)

src_folder = "./data/src"
subsets = ["train", "dev", "test"]

for subset in subsets:
    text_file_path = os.path.join(src_folder, subset, "text")
    wav_scp_file_path = os.path.join(src_folder, subset, "wav.scp")
    output_wav_scp_file_path = os.path.join(src_folder, subset, "wav.scp")

    print(f"Filtering {subset} subset...")
    filter_wav_scp(text_file_path, wav_scp_file_path, output_wav_scp_file_path)
    print(f"{subset} subset filtered wav.scp saved to {output_wav_scp_file_path}")
