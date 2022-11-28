[English](./cuside.md) | [中文](./cuside_ch.md)

## Streaming Speech Recognition

Streaming speech recognition refers to perform recognition simultaneously when the speaker is speaking, rather than waiting for the speaker to complete a sentence before starting recognition.

However, the neural network architectures commonly used at present, such as the **transformer** and **conformer**, are based on the self-attention mechanism, and thus rely on the entire sentence as input. So they are not directly suitable for low-latency speech recognition.

In order to solve this problem, many systems adopt **chunk models**. Specifically, a sentence will be divided into multiple chunks, and then sent to the neural network for recognition chunk by chunk. In this way, the latency is reduced to about the length of one chunk.

## Context Sensitive Chunk

In chunk-based low-latency speech recognition models, a common practice is to attach certain historical frames and future frames to each chunk to provide context information and form a **context sensitive chunk**.

Existing work has shown that context information is crucial to accurate acoustic modeling, and the lack of context information will cause more than 10% loss of recognition accuracy.

However, in order to get future information, the model has to wait until a certain number of future frames arrive before starting recognition, which significantly increases the recognition latency.

To solve this problem, we propose a new framework - **Chunking, Simulating Future Context and Decoding (CUSIDE)** for streaming speech recognition.

## CUSIDE

The CUSIDE model uses simulated future frames instead of real future frames for recognition, and thus reduce reliance on future information and reduce recognition delays.

The **simulation frames** are generated in a streaming manner using a **simulator**, which consists of a **simulation encoder** and a **simulated predictor**. The simulator can be trained in an unsupervised manner, because the target future frame can be obtained by shifting the real future frame backward. Here we are inspired by the unsupervised representation learning method (APC), which does not require additional annotation information.

In addition, we unify streaming and non-streaming model by weight sharing and joint training (called unified streaming/non-streaming model) to reduce performance gap between streaming and non-streaming models. We obtain new state-of-the-art streaming ASR results on the AISHELL-1 dataset (4.79% CER).

The newly-added code to support CUSIDE can be found at: `cat/ctc/train_unified.py` for CTC/CTC-CRF, `cat/rnnt/train_unified.py` for RNNT, respectively.

## Model hyper-parameters

Training hyper-parameters are configured in `trainer` in `config.json`.

- `downsampling_ratio`: the downsampling rate before and after the encoder, default **8**.
- `chunk_size`: chunk_size (frames),default **40**.
- `context_size_left`: left context frame number, default **40**.
- `context_size_right`: right context frame number, default **40**.
- `jitter_range`: chunk size jitter range, default **2**.  Note that `jitter_range` denotes the jitter range after encoder output, which amounts to `jitter_range`*`downsampling_ratio` (frames) in the encoder input.
- `mel_dim`: Mel-spectrogram dimension, default **80**,
- `simu`: simulated future frames. By default, the simulation encoder is a uni-directional GRU and the predictor is a simple feed-forward neural network.
- `simu_loss_weight`: The weight for the simulation loss (**L1 loss** by default),  i.e., \alpha in Algorithm 1 in the CUSIDE paper.

```python
    "trainer": {
        "compact": true,
        "downsampling_ratio": 8,
        "chunk_size": 64,
        "simu": true,
        "context_size_left": 64,
        "context_size_right": 32,
        "jitter_range": 4,
        "mel_dim": 80,
        "simu_loss_weight": 100
    }
```

## Training

- In training, streaming model and non-streaming model have shared parameters and are jointly trained. For each training utterance, the streaming and non-streaming losses are calculated respectively and summed together as show in Algorithm 1 in the CUSIDE paper.
- `chunk_forward()` is the function, performing forward calculation in the streaming manner **in training**; `chunk_infer()` is the function, performing forward calculation in the streaming manner **in testing**. The difference between the two functions is as follow:
   - `chunk_forward()` is used **in training**, which uses **chunk size jitter** and **stochastic future context**. Stochastic future context means that the future context is randomly chosen from three settings, a) using simulated future frames; b) no future frames; c) using real future frames. 
   - `chunk_infer()`  is used **in testing**, which uses a fixed chunk size and, in experiments, employs a fixed setting for future context, from one of the three settings introduced above.
- The key operations in both `chunk_forward()` and `chunk_infer()`  are dividing chunks and splicing context frames to each chunk. 
   - Dividing chunks is realized  by reshaping of the tensors.
   - Splicing context frames is obtained by shifting the sequence of chunks to the left or the right according to the relationship between the context and the current chunk. For example, consider the setting where the left context, chunk size, and right context are all of 40 frames. Then, the left context of the current chunk is actually the previous chunk, while the right context of the current chunk is actually the next chunk. If the simulated future frames are used, the right context of the current chunk is obtained by feeding the current chunk to the simulator.

- In training, the future context of all chunks in an utterance are simulated at one time, instead of simulating chunk by chunk.
- The `forward()` function also contains the code to perform forward calculation in the non-streaming manner, which is taken the same as in a standard non-streaming model.

## Decoding

- Decoding setting is configured by `infer` in `hyper-p.json`, where `unified` is required to be true. `streaming`  determines whether streaming or non-streaming decoding is executed.

```python
"infer": {
    "bin": "cat.rnnt.decode",
    "option": {
        "unified": true,
        "streaming": true,
        "beam_size": 16,
        "cpu": true,
        "nj": 40,
        "resume": "exp/best-10.pt"
        }
    }
```

## References

- [CUSIDE: Chunking, Simulating Future Context and Decoding for Streaming ASR.](http://oa.ee.tsinghua.edu.cn/ouzhijian/pdf/cuside-intespeech2022-camera.pdf)