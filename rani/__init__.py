"""
rani-activation
===============
RANI: Rational Adaptive Normalizing Initialization

PyTorch implementation. TensorFlow support coming when TF adds Python 3.14.

Quick start:
    from rani import RANI             # Module with learnable params
    from rani import rani             # Functional with fixed params
    from rani import init_rani_network  # Correct weight initialization

Author  : Nikki Rani
Version : 1.0.1
"""

from rani.torch_rani import (
    RANI,
    rani,
    lecun_normal_,
    init_rani_network,
)

__version__ = "1.0.0"
__author__  = "Nikki Rani"
__all__     = [
    "RANI",
    "rani",
    "lecun_normal_",
    "init_rani_network",
]
