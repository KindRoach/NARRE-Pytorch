from dataclasses import dataclass

import torch


@dataclass
class BaseConfig(object):
    num_epochs: int
    batch_size: int
    learning_rate: float
    l2_regularization: float
    learning_rate_decay: float
    device: str


class BaseModel(torch.nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.current_epoch = 0
        self.train_loss = dict()
        self.config = config

    def get_device(self) -> torch.device:
        return list(self.parameters())[0].device
