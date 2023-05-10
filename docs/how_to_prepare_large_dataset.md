[English](./how_to_prepare_large_dataset.md) | [中文](./how_to_prepare_large_dataset_ch.md)

# Processing, preparation and training with large datasets

## Background

On small data sets, the ordinary **audio data processing** consists of two stages:

1. _Preprocessing (feature extraction)_: we use kaldi/torchaudio to generate `.ark/.scp` files, in which the `.ark` file is the binary file that stores the feature, the `.scp` file is the index of the `.ark` file, and there is also a text file as the corresponding transcription for the audio;
2. _Data preparation_: We pack the data into a format that is easy for python to read (refer to [code](../cat/utils/pipeline/asr.py#L20) for specifics), which includes the following three operations. First, save the feature sequence lengths for subsequent dynamic batching; Second, pad the label sequences (encoded into numbers by tokenizer) so that they can be saved in `numpy.ndarray` format; Third, save the index corresponding to the features (similar to the `.scp` files). The final saved file is the package of the above several files.

In the process outlined above, the major time cost is the data preprocessing (Stage 1), while the time cost of Stage 2 is very small. The Stage 2 for 1000-hour data only takes a few minutes (limited to hard disk IO).

When using the processed data in **model training**, data loading is developed based on the standard `Dataset`  (map-style) interface of `torch`, which has the following two features:

1.  After setting `shuffle = True` , data loading is **completely random**: Any two sentences can appear in the same mini-batch. It is well known that the randomness of sentences in mini-batches is needed for the stochastic optimization of neural network models.
2.  The data is read by **the indexing method** (rather than one-time loading of all the data from the hard disk into memory). However, thanks to the memory cache mechanism of the OS level, if the memory is large enough, after one cycle (one epoch), all the data will be loaded into the memory in a transparent way (not concretely perceived by users). In the subsequent training, the data will in fact be read directly from the memory, rather than from the hard disk. So although the above Feature 1 will lead to a large amount of random reading and writing,  the speed is still very fast, since the reading and writing is taken over the memory.

In the above discussion, we actually ignore one problem: **what if the memory is not enough, or alternatively say, the dataset is pretty large?** This question leads us to realize the following drawback of the data loading method outlined above.

When the memory is not enough to load the entire dataset, loading all the data into the memory before the training starts will incur out-of-memory, which will invoke SIGKILL to kill the process. By the use of the indexing method for data loading, the training can still be carried out normally. However, something changes on the OS level. When the system memory is full and the new data to be read is not in the memory, the OS will clean up some data in the memory to provide space for the new data. In this process, the data loading is `hard disk --> memory`, which is very slow (compared with `memory -->CPU/GPU`). Since model training often requires multiple iterations, each round of slow data loading will lead to a waste of time. Especially when `data size >> memory size`, it is almost equivalent to reading data directly on the hard disk. The random reading/writing speed from the hard disk is lower (For speed comparison, **sequential memory access >> random memory access > sequential disk access >> random disk access**).  So the users will clearly observe that the time cost of training iteration increases almost exponentially with the increase of training data, This is unacceptable for the training with large data sets.

## Scheme

For many reasons, we cannot easily expand our hardware to support larger and larger data sets (in fact, about 1200 hours of 80-dim FBank data can occupy 256 GB of memory). Therefore, reading data from the hard disk (rather than from the faster memory) is an inescapable problem. Note that the sequential reading speed from the hard disk is far greater than the random reading performance. We can start from the above Feature 1 and modify the data loading.

### Solution provided by [webdataset](https://github.com/webdataset/webdataset)：

The idea is to reduce the randomness of data loading. Completely sequential reading (i.e., removing randomness completely) will reduce the performance, but we can take a trade-off: dividing the entire dataset into a set of `tar` files (called **sharding**). Each `tar` file stores features for several utterances (for example, 2000). A `tar` file can be also called an ark-list, since an `ark` file contains the feature for one utterance. At the beginning of each epoch, **shuffle at the tar level, and shuffle again at the utterance level within each tar.** This scheme can retain certain randomness, and at the same time,  can also reduce random access. The ark files in a `tar` file are sequentially read, which can improve IO performance; and further, they can all be loaded into memory. The shuffle within the `tar` can then be taken all within memory, which is fast.

Based on `webdataset`, when processing large datasets (depending on the memory size, generally more than 1500 hours), we revise the data processing as follows:

1. Perform data preprocessing as in the ordinary Stage 1.
2. Pack the features and the labels (in text format) into `tar` files: every 2000 sentences into a `tar` file. This process does not involve calculation, but mainly involves a lot of IO operations.

Step 1 & 2 can be executed respectively or simultaneously. Please check the example of usage for details.

**NOTE:**
A non-trivial difference between the new scheme and the ordinary method is that **the label will be saved in text format**. In model training, the tokenizer can directly performs the on-the-fly encoding when loading the data. The additional overhead is negligible.

In particular, when using some tokenizers, you should take care to handle the spaces in the label appropriately, for example:

For SentencePiece tokenizer using Chinese characters (tokenizer training without spaces), if the spaces in the label are not removed when the data is prepared, they will be mapped to `<unk>`, which will seriously affect the model performance. Therefore, for Chinese datasets, it is better to remove the spaces in the label before data sharding. For some tokenizers that are insensitive to spaces (such as Jieba word-segmentation tokenizer), spaces do not affect word segmentation, so removing spaces does not matter.

## Example of usage

Please refer to the experiment [yesno](../egs/TEMPLATE/exp/asr-ctc-large-corpora)


**NOTE:**

- Since the dev set is always configured as `shuffle=False`, and generally small to fit into memory, we keep the dataloading of dev set in the ordinary way.
- The normal concept of **epoch** does not exist, for data is now coming as a stream. Therefore, in model training, the epoch id in output log will always be 1. Though we cannot obtain the number of training epochs in a strict manner, one can make a roughly estimate by:

   ```
   num_epochs = num_steps * batch_size / num_total_utts
   ```

- Continuing from a stopped training (`--resume`) is not a strict resuming, and would inevitably introduce biases to part of the data (this should be negligible to the overall training if not stop & resume frequently).

## References

1. [webdataset/webdataset: A high-performance Python-based I/O system for large (and small) deep learning problems, with strong support for PyTorch. (github.com)](https://github.com/webdataset/webdataset)
2. [wenet/UIO.md at main · wenet-e2e/wenet (github.com)](https://github.com/wenet-e2e/wenet/blob/main/docs/UIO.md)
3. [Distributed Training with Uneven Inputs Using the Join Context Manager — PyTorch Tutorials 1.10.1+cu102 documentation](https://pytorch.org/tutorials/advanced/generic_join.html#how-does-join-work)