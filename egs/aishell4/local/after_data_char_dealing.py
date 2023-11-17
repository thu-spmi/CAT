import os

"""
函数功能说明：在处理完text文件后，删除wav.scp文件中对应的行数
"""

def filter_wav_scp(text_file_path, wav_scp_file_path, output_wav_scp_file_path):
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
