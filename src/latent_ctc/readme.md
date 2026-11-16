# Latent CTC

A PyTorch CUDA extension implementing the **Latent CTC** loss: a CTC criterion
whose target sequence may mix two kinds of labels

* **subclass** ids in `[0, V)` — supervised as usual, one class per position;
* **major-class** ids encoded as `major_label_offset + major_index` — scored by
  marginalizing every subclass registered for that major class.

The network output stays the subclass vocabulary of size `V`; the major classes
have no output neurons of their own. This is the loss behind the Latent Softmax
output layer, where tone-marked vowels are subclasses and base vowels are major
classes.

The kernels are derived from `warp-ctc` (SeanNaren) and CTC-CRF (Tsinghua
University). Only the CTC / Latent-CTC part is kept here: the CTC-CRF
denominator and its OpenFST dependency are **not** included.

## Where it is used

`cat/ctc/train_latent_softmax.py`, when `trainer.loss_type` is `latent_ctc`
(the default). Recipe experiments:

| experiment | loss in the released config |
| :--- | :--- |
| `egs/latent-softmax/multi_exp/21_latent_softmax_120M_jnl_lossfn` | `latent_ctc` |
| `egs/latent-softmax/cs_exp/06_ASRU19_latent_S2P_adjust_lr` | `latent_ctc` |
| `egs/latent-softmax/cs_exp/12_CS_Dialogue_latent_S2P_adjust_lr` | `latent_ctc` |

Import path: `from latent_ctc import LATENT_CTC_LOSS`.

> The class used to be shipped inside the CTC-CRF extension and called as
> `ctc_crf.LATENT_CTC_LOSS`. It now lives in its own package and is called as
> `latent_ctc.LATENT_CTC_LOSS`.

## Usage

```python
import torch
from latent_ctc import LATENT_CTC_LOSS

criterion = LATENT_CTC_LOSS(
    latent_map=[[2, 3], [4, 5]],   # major 0 -> subclasses 2,3 ; major 1 -> 4,5
    major_label_offset=6,          # first id used for major-class labels
    blank_label=0,
)

loss = criterion(logits, labels, input_lengths, label_lengths)
```

| argument | shape / dtype | notes |
| :--- | :--- | :--- |
| `logits` | `(N, T, V)` float32, CUDA | log-probabilities, i.e. after `log_softmax` |
| `labels` | `(sum(U_i),)` int32, **CPU** | flattened without padding; `< V` are subclasses, `>= major_label_offset` are majors |
| `input_lengths` | `(N,)` int32, **CPU** | |
| `label_lengths` | `(N,)` int32, **CPU** | |

`latent_map` accepts a list of subclass-id lists, a JSON path, or a dict with
`major_to_subclasses` (or `majors`) plus an optional `major_label_offset`. The
recipe generates it with
`egs/latent-softmax/local/majorClass2subClass.py --subclass-list-out`, which
writes the same nesting as `TonalLexiconTokenizer.get_subClassList`.

## Build and install

Requirements:

* Linux with a working `nvcc` (the `Makefile` defaults to `CUDA_HOME=/usr/local/cuda-12.1`)
* `cmake`, `gcc`/`g++`, `python3` with `pip`
* PyTorch built against the same CUDA version (`torch>=1.12`)
* `ninja` (`pip install ninja`) — `setup.py` builds the extension with `use_ninja=True`

Run from the repository root:

```bash
cd src/latent_ctc
make
```

`make` runs two steps:

1. `make GPUCTC` — CMake builds the CUDA library into `gpu_ctc/build/libwarpctc.so`;
2. `make CTCCRF` — `python3 setup.py install` builds and installs the Python
   binding **into the currently active Python environment**.

If your toolkit is not at `/usr/local/cuda-12.1`, pass it explicitly:

```bash
make CUDA_HOME=/usr/local/cuda-12.4
```

`make verify` prints the CUDA / GPU / PyTorch versions it will build against;
`make help` lists all targets.

> The defaults target `sm_90` (H100): `H100_ARCH` feeds
> `-DCMAKE_CUDA_ARCHITECTURES` for the CMake step, and `setup.py` passes
> `-gencode=arch=compute_90,code=sm_90` for the binding. For another GPU, change
> both — the Makefile variable alone is not enough.

Verify after installing:

```bash
python -c "import latent_ctc, torch; print(latent_ctc.__version__, torch.cuda.is_available())"
```

## Verified hardware

The extension has only been validated on:

* NVIDIA H100 (sm_90)
* NVIDIA L40S (sm_89)
* NVIDIA RTX 3090 (sm_86)
* NVIDIA RTX 4090 (sm_89)

Other GPUs and other CUDA / PyTorch combinations have **not** been tested.

## License

Apache 2.0. The CUDA kernels are derived from `warp-ctc` (Copyright 2016
SeanNaren) and CTC-CRF (Copyright 2018-2022 Tsinghua University); the
`gpu_ctc/contrib/moderngpu` headers keep their own license, see
`gpu_ctc/contrib/moderngpu/LICENSE`.
