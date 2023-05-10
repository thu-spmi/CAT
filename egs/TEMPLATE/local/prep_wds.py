"""
Prepare data for training very large corpora.

Usage:
    1. download and process data first
        bash local/data.sh

    2. run this script
"""

from cat.utils.data.prep_wds import WdsSink, FeatExtractor, pack_data_audio, pack_data_ark

if __name__ == "__main__":
    d_out = f"data/wds"
    pattern = "%05d.tar"

    # create a data writer
    writer = WdsSink(
        d_out,
        pattern,
        # group data according to frame lengths
        filters=[
            (0, 10),
            (10, 1000),
            (1000, 1200),
            (1500, 2000),
            (2000, 3000),
            (3000, -1)
        ],
    )

    # example of packing data from audios
    # yesno is sampled by 8kHz. Generally the sampling rate is 16000
    feat_reader = FeatExtractor("data/src/yesno/wav.scp", sample_rate=8000, kaldi_style=True)
    pack_data_audio(feat_reader, writer, "data/src/yesno/text")

    # example of packing data from kaldi .ark file
    # pack_data_ark(writer, "data/src/yesno/feats.scp", "data/src/yesno/text")
