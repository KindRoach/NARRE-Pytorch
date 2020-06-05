from dataclasses import dataclass
from typing import List

import torch

from model.base_model import BaseModel, BaseConfig


@dataclass
class NarreConfig(BaseConfig):
    pad_word_id: int = 3000000
    pad_item_id: int = 4999
    pad_user_id: int = 4999
    user_count: int = 5000
    item_count: int = 5000
    review_length: int = 200
    review_count: int = 25
    word_dim: int = 300
    kernel_width: int = 5
    kernel_deep: int = 100