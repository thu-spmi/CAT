# CUSIDE-Array

## Introduction
This project is based on the research presented in the paper titled [A Streaming Multi-Channel End-to-End Speech Recognition System with Realistic Evaluations](https://arxiv.org/abs/2407.09807).

More detailed introduction to Streaming multi-channel end-to-end (ME2E) ASR in Chinese can be found in [ME2E_ASR_ch](ME2E_ASR_ch.md).

Experiments on AISHELL-4 can be found [here](../egs/aishell4/README.md).

## Background

Speech recognition technology enables computers to understand and process human speech. Multi-channel automatic speech recognition (ASR) systems, which use multiple microphones, can improve the accuracy and robustness of speech recognition, especially in noisy environments.

Recently, multi-channel end-to-end (ME2E) ASR systems have been developed. These systems combine the front-end, which processes the multi-channel audio signals, and the back-end, which recognizes the speech, into one unified model.

## Main Contribution

#### CUSIDE-Array:

This method integrates the CUSIDE methodology (Chunking, Simulating Future Context, and Decoding) into the ME2E ASR system. It processes speech in small chunks and simulates future context to improve accuracy while keeping the system fast.

The total latency (delay in processing) is 402ms, making it suitable for real-time applications.

#### OOD & ID Evaluations:

Recent studies call attention to the
gap between ID and OOD tests, and the importance of **realistic evaluations**, which mean conducting both ID and OOD
testings.

ID (In-Distribution): Tests performed on data that is similar to the training data.

OOD (Out-of-Distribution): Tests performed on data that is different from the training data. 

OOD generalization is crucial because it shows how well the system performs in real-world scenarios where the speech environment can vary.

## How It Works

### Context-Sensitive Chunking:

The audio input is divided into small chunks.
Each chunk is processed with additional frames from before and after to provide context, improving the accuracy of recognition.

### Array Beamforming:

The front-end uses a mask-based MVDR neural beamformer to enhance the speech signal from the multi-channel input.
This helps in reducing noise and focusing on the main speech signal.

### Future Context Simulation:

A neural network simulates the future context of the speech signal.
This simulation helps in making better predictions without having to wait for the actual future frames, thus reducing latency.

