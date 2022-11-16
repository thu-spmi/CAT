# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""LM-related modules
"""

from .train import build_model as lm_builder

__all__ = [lm_builder]
