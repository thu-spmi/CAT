import ctc_crf
import torch

den_lm = "den_lm.fst"

# 0: <blk>
# 1: a
# 2: c
# 3: s
# 4: t
vocab_size = 5


def test():
    criterion = ctc_crf.CTC_CRF_LOSS(lamb=0.01)
    logits = torch.tensor([
        [
            [0.1, 0.1, 0.5, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.2],
            [0.1, 0.7, 0.1, 0.05, 0.05],
            [0.6, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.6, 0.1]
        ]
    ], device=0, dtype=torch.float32, requires_grad=True).log()
    # [2, 1, 4] -> c a t
    labels = torch.tensor([2, 1, 4], dtype=torch.int32)
    frame_lens = torch.tensor([5], dtype=torch.int32)
    label_lens = torch.tensor([3], dtype=torch.int32)
    print("Frame len: {}".format(frame_lens.tolist()))
    print("Label len: {}".format(label_lens.tolist()))
    print("Logit shape: {}".format(logits.shape))
    print("Label shape: {}".format(labels.shape))

    loss = criterion(logits, labels, frame_lens, label_lens)
    print("CRF loss:", loss.item())

    loss.backward()


if __name__ == "__main__":
    ctx = ctc_crf.CRFContext(den_lm, gpus=0)
    test()
