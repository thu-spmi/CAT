
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str, help="path of data dir")
args = parser.parse_args()
file_name = os.path.join(args.data_dir, "utt2dur")

assert os.path.isfile(file_name), "this script require utt2dur for calculate total duration."

# start_time = time.time()
total_duration = 0.
with open(file_name, "r") as f:
    for line in f:
        path = line.split()[1]
        duration = float(path)
        # duration = librosa.get_duration(filename=path)
        total_duration += duration
# end_time = time.time()
print(f"total duration: {total_duration/3600:2f} hour")
# print(f"process time : {end_time-start_time:2f} second")



