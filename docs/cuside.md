## Streaming Speech Recognition

Streaming speech recognition refers to recognition while the speaker is speaking, rather than waiting for the speaker to speak a complete sentence before starting recognition.

However, the neural network structures commonly used in the industry at present, such as the **transformer** and **conformer** based on the self-attention mechanism, all rely on the entire sentence as input, so they are not suitable for low-latency speech recognition.

In order to solve this problem, many systems adopt a `chunk model`. Specifically, a sentence will be divided into multiple blocks, and then sent to the neural network for recognition block by block, thus reducing the delay to the length of one block.

## Context Sensitive Chunk

In block-based low-latency speech recognition models, a common practice is to attach certain historical frames and future frames to each block to provide context information and form a context sensitive chunk.

Existing work has shown that context information is crucial to accurate acoustic modeling, and the lack of context information will cause more than **10%** loss of recognition accuracy.

However, in order to get future information, the model has to wait until a certain number of future frames arrive before starting recognition, which significantly increases the recognition latency.

To solve this problem, we propose a new framework - **Chunking, Simulating Future Context and Decoding** (CUSIDE) for streaming speech recognition.

## CUSIDE

In the CUSIDE model, the model using simulated future frames instead of real future frames for recognition,and reduce reliance on future information and reduce recognition delays.

The simulation frame is generated in a streaming manner using a generator, which consists of a generator encoder and a generator predictor, which can be trained in an unsupervised manner (because the corresponding prediction target can be obtained by moving the input frame forward, here We are inspired by the unsupervised representation learning method APC), which does not require additional annotation information.

Except,streaming and non-streaming model weight sharing and joint training (unified streaming/non-streaming model)to reduce performance between streaming and non-streaming models, we obtain new state-of-the-art streaming ASR results on the AISHELL-1 dataset industry-leading results.

## Parameters Description

Main code at `cat/rnnt/train_unified.py`,Parameters:

- `config.json` **trainer** configure the following parameters

  - `downsampling_ratio`: Input the downsampling rate before and after the encoder, default **8**.
  - `chunk_size`: chunk_size(frames),default **40**.
  - `context_size_left`: left context frame,default **40**.
  - `context_size_right`: right context frame,default **40**.
  - `jitter_range`: chunk size jitter range,default **2**. 
  - `mel_dim`: Mel-spectrograms dim,default **80**,
  - `simu`: simulated future frames,encoder GRU and predictor is a simple feed-forward neural network,defaul **false**.
  - `simu_loss_weight`: A weighting factor for the composite loss. Default **L1 loss**.

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

## Training process

- Training process, streaming model and non-streaming model perform parameter sharing and joint training.For training data,streaming and non-streaming get loss in order.
- `chunk_forward()` for training forward by streaming `chunk_infer()` for testing forward by streaming.In differ  `chunk size jitter` used during training and the use of random future information (future information is randomly selected from a. simulated future frames; b. w/o future frames; c. real future frames),but for testing `chunk_size` fixed and one of the three cases is fixed according to the setting.
- `chunk_forward()` and `chunk_infer()` divide chunks and stitch context for each chunk.Dividing chunks is achieved by reshaping,and the context is obtained by shifting the sequence of chunks to the left or right through the relationship between the context and the current chunk: for example, consider the case where the left context, chunk size, and right context are all 40, then the current chunk The left contxt of the current chunk is actually the previous chunk, and the right context of the current chunk is actually the next chunk. If the simulated future frame is used, the right context of the current chunk is the output after the simu net receives the current chunk.
- During training, we simulate the future context of all chunks at one time, instead of simulating block by block.
- `forward()` non-streaming is the same as training a standard non-streaming model, and will not be repeated here.

## Decoding

- For decoding `unified` true is required,`streaming` parameter controls streaming and non-streaming decoding.

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