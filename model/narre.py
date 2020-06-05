from dataclasses import dataclass
from typing import List

import torch

from model.base_model import BaseModel, BaseConfig


@dataclass
class NarreConfig(BaseConfig):
    review_length: int
    review_count: int
    word_dim: int  # the dimension of word embedding
    kernel_width: int  # the window size of convolutional kernel
    kernel_deep: int  # the number of convolutional kernels
    latent_factors: int
    fm_k: int
