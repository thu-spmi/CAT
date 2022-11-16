
import sys
import torchaudio
from tqdm import tqdm
from kaldiio import WriteHelper

if __name__ == "__main__":
    wavin = sys.argv[1]
    arkout = sys.argv[2]
    if wavin == '-':
        wavin = '/dev/stdin'
    if arkout == '-':
        arkout = '/dev/stdout'

    if wavin != '/dev/stdin':
        tot = sum(1 for _ in open(wavin, 'r'))
    else:
        tot = None

    with open(wavin, 'r') as fi, WriteHelper(arkout) as writer:
        for line in tqdm(fi, total=tot):
            uid, file = line.strip().split(maxsplit=1)
            waveform = torchaudio.load(file)[0].view(-1, 1)
            writer(uid, waveform.numpy())
