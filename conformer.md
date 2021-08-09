# Conformer

We implement the Conformer architecture basically according to [TensorFlowASR](https://github.com/TensorSpeech/TensorFlowASR) and [Huggingface](https://github.com/huggingface/transformers).

* For implementation of **positional encoding**, **convolution subsampling**, **relative multi-head attention**, **feed-forward module**, **convolution module** and **multi-head attention module**, please refer to `scripts/ctc-crf/_layers.py`.
* For top interface of Conformer model, please refer to the `ConformerNet` in `scripts/ctc-crf/model.py`
* For results on swbd, librispeech, CommonVoice German, refer to ["Advancing CTC-CRF Based End-to-End Speech Recognition with Wordpieces and Conformers"](https://arxiv.org/abs/2107.03007).

## References

* Anmol Gulati, James Qin, Chung-Cheng Chiu, et al., “Conformer: Convolution-augmented Transformer for speech recognition,” in INTERSPEECH, 2020, pp. 5036–5040.
* TensorFlowASR: https://github.com/TensorSpeech/TensorFlowASR
* Huggingface: https://github.com/huggingface/transformers

