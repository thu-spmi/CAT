# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""CTC-related modules
"""


from .train import build_model as ctc_builder

__all__ = [ctc_builder]
