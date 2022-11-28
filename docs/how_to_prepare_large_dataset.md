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

At present, in order to be compatible with the ordinary method, Stage 1 and 2 are separated. In future, Stage 1 and 2 can be combined to further improve the efficiency of feature processing.

**NOTE:**
A non-trivial difference between the new scheme and the ordinary method is that **the label will be saved in text format**. This is because we may change the tokenizer in practical experiments. If we save the IDs of the label, we need to run Stage 2 again, which is not worth doing it. After saving the text information of the label, the tokenizer can directly performs the on-the-fly encoding when loading the data. The additional overhead is negligible.

In particular, when using some tokenizers, you should take care to handle the spaces in the label appropriately, for example:

For SentencePiece tokenizer using Chinese characters (tokenizer training without spaces), if the spaces in the label are not removed when the data is prepared, they will be mapped to `<unk>`, which will seriously affect the model performance. Therefore, for Chinese datasets, it is better to remove the spaces in the label before data sharding. For some tokenizers that are insensitive to spaces (such as Jieba word-segmentation tokenizer), spaces do not affect word segmentation, so removing spaces does not matter.

In future, we may consider separate processing of audio features and text labels. The text processing is relatively easy, and one-time full loading into memory will not bring too much overhead.

## Interface design

### Data preparation
The code to implement Stage 2 using `webdataset` can be found in [code](../egs/wenetspeech/local/prep_wds.py#L16). The function interface is:

```python
# Number of sentences saved in each file, no need to modify without special needs
UTTS_PER_FILE = 2000

def pack_data(
        # scp index files in kaldi format. Multiple files can be input as a list
        f_scps: Union[List[str], str],
        # Text files. The first column is the sentence ID, which should match the ID in the scp file. Multiple files can be input as a list
        f_labels: Union[List[str], str],
        # output folder
        d_out: str,
        # format of output file
        fmt: str = "data-%05d.tar",
        # Configuration of length grouping. For example, "10:2000" means that only sentences with lengths of 10-2000 frames are reserved. Multiple groups can be used,
        # like，["10:500", "500:800", "800:1000", "1000:1200"], files from different length groups will be saved in corresponding folders
        filter_group: List[str] = None):
    ...
```

### Model training

In the `hyper-p.json` file, set `train:option:large_dataset = True`, and set `train:option:tokenizer=xxx` and `train:option:trset=xxx`. `trset` specifies the format of output file. For example:

```
# During data processing, if specifying d_out='./data', filter_group=['10:1000', '1000:2000']
# Then trset can be specified as
# 1. only use sentences of length 10-1000
trset='./data/10_1000/data-*.tar'
# 2. use sentences of length 10-2000
trset='./data/{10_1000,1000_2000}/data-*.tar'
# 3. code debug, only use 10x2000 sentences
trset='./data/10_1000/data-0000{0..9}.tar'
```

The underlying `webdataset` interface call is implemented in [code](../cat/shared/manager.py#L79).

**NOTE:**  The development data can be configured as  `shuffle = False`. Since the amount of development data is generally small, the development data can still be loaded in the ordinary way.

## DDP (Distributed Data Parallel)

All the above discussions are based on the case of single-machine and single-card. When DDP multi-card or multi-machine and multi-card training mechanism is involved, this problem will become a little bit tricky. For example:

```
trset="data-0{0..2}.tar"    # Contains three tar files, with a total of 3x2000 sentences
# Suppose that there are two processes (two GPUs) using DDP training,
# do shuffle at the tar level, and assign the ark file to the two processes after shuffle
gpu0: data-00.tar
gpu1: data-02.tar, data-01.tar
```

The consequent problem with the above code is that the amount of data on the two processes are different. Note that DDP is synchronous gradient update in training. In direct running of the above code, gpu1 will always wait for gpu0 to synchronize, but gpu0 has finished the transversal of all data and exited. The solution proposed by [wenet-e2e](https://github.com/wenet-e2e/wenet/blob/main/docs/UIO.md#qa) to address this problem is to use `model.join()`.

In contrast, we use a simpler and more direct manner. When a process finish traversing all its data, it directly forces all the processes to stop the current round (epoch). In this way, the amount of data trained in one epoch is reduced. However, we iterate for a number of epochs and shuffle at the `tar` level, the impact of this manner is relatively small.

> wenetspeech L (~10,000 hour) contains about 15 million sentences, which are processed to yield about 7500 tar files. Training with 8 GPU, 7500% 8=4, that is, 4x2000 sentences are discarded in each epoch.

**NOTE:**

The above example is just for ease of understanding. In practice, webdataset will re-use some of the `.tar` files to ensure the number of the tar files being evenlt distributed over cards. However, because the number of utterances in a dataset may not be exactly divided by 2000, there may have fewer sentences in one (or multiple) `tar` files than others. In the worst situation, `2000*(N-1)` utterances are discarded each epoch (N denotes \#GPUs).

## References

1. [webdataset/webdataset: A high-performance Python-based I/O system for large (and small) deep learning problems, with strong support for PyTorch. (github.com)](https://github.com/webdataset/webdataset)
2. [wenet/UIO.md at main · wenet-e2e/wenet (github.com)](https://github.com/wenet-e2e/wenet/blob/main/docs/UIO.md)
3. [Distributed Training with Uneven Inputs Using the Join Context Manager — PyTorch Tutorials 1.10.1+cu102 documentation](https://pytorch.org/tutorials/advanced/generic_join.html#how-does-join-work)