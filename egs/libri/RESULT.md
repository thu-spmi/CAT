# Librispeech

Results on Librispeech dataset.

## Conformer+Transformer rescoring

* Reported in ["Advancing CTC-CRF Based End-to-End Speech Recognition with Wordpieces and Conformers"](https://arxiv.org/abs/2107.03007)
* AM: Conformer with 52M parameters. SpecAug is applied.
* "Trans." in the table denotes the Transformer (indeed the interpolation with 4-gram).

| Unit  | LM     | Test-clean | Test-other |
| ----- | ------ | ---------- | ---------- |
| phone | 4-gram | 3.61       | 8.10       |
| phone | Trans. | 2.51       | 5.95       |
| wp    | 4-gram | 3.59       | 8.37       |
| wp    | Trans. | 2.54       | 6.33       |

**Experiment**

* Phone-based system

  ```shell
  bash run.sh
  ```

* Char-based or wordpiece-based

  ```shell
  bash run_wp.sh
  ```

**To reproduce our Trans. LM rescoring, you need to:**

1. Download and configure `returnn` from https://github.com/rwth-i6/returnn.
2. Download the 42-layer pretrained word-level Trans. LM from [1], and put `2019-lm-transformers/*` under `returnn/` directory.
3. Modify the path in `conf/rwth-nnlm.conf` and `local/pytorchnn/lmrescore_nbest_pytorchnn_rwth.sh` accordding to your systems.
4. Copy `local/pytorchnn/lmrescore_nbest_pytorchnn_rwth.sh` to `steps/pytorchnn/`
5. Replace `returnn/2019-lm-transformers/librispeech/word_200k_vocab/re_transfo_42_d00.sgd.lr1.cl1.small_batch.config` with `conf/rwth-nnlm.conf`

**Reference**

[1] https://github.com/rwth-i6/returnn-experiments/tree/master/2019-lm-transformers