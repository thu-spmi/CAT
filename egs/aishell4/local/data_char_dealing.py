# Copyright 2020 Tsinghua SPMI Lab / Tasi
# Apache 2.0.
# Author: Xiangzhu Kong (maxwellzh@outlook.com)
#
# Description:
#   This script processes text files by removing all strings matching a specific pattern '<*>'.
#   The key functions include collecting special strings from the text file, backing up the original text file, 
#   and writing the cleaned text back to the file.

import os
import re
import shutil

def collect_special_strings(file_path):
    """
    Collects special strings matching the pattern '<*>' from the given file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        dict: A dictionary with the special strings as keys and the corresponding uids as values.
    """
    special_strings = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            uid, text = line.strip().split("\t")
            matches = re.findall(r'<(.*?)>', text)
            for match in matches:
                if match in special_strings:
                    special_strings[match].append(uid)
                else:
                    special_strings[match] = [uid]

    return special_strings

src_folder = "./data/src"
subsets = ["dev", "test", "train"]
text_filename = "text"
backup_filename = "text.bak"

for subset in subsets:
    subset_folder = os.path.join(src_folder, subset)
    text_file_path = os.path.join(subset_folder, text_filename)
    backup_file_path = os.path.join(subset_folder, backup_filename)

    shutil.copy(text_file_path, backup_file_path)

    special_strings = collect_special_strings(text_file_path)

    with open(text_file_path, "w", encoding="utf-8") as f:
        for line in open(backup_file_path, "r", encoding="utf-8"):
            uid, text = line.strip().split("\t")
            cleaned_text = re.sub(r'<.*?>', '', text)  # Remove special strings
            if cleaned_text.strip():  # Check if text is not empty after cleaning
                f.write(f"{uid}\t{cleaned_text}\n")

    print(f"Collected unique special strings in {subset}:")
    for idx, (special_string, uids) in enumerate(special_strings.items(), start=1):
        uids_str = ", ".join(uids)
        print(f"<*{idx}>:{uids_str};")
