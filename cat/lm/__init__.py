# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""LM-related modules
"""

from .train import build_model as lm_builder

__all__ = [lm_builder]
