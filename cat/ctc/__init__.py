# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""CTC-related modules
"""


from .train import build_model as ctc_builder

__all__ = [ctc_builder]
