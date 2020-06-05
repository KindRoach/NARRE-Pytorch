from dataclasses import dataclass

import torch


@dataclass
class BaseConfig(object):
    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    l2_regularization: float = 1e-6
    learning_rate_decay: float = 0.99
    device: str = "cpu"


class BaseModel(torch.nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.current_epoch = 0
        self.train_loss = dict()
        self.config = config

    def get_device(self) -> torch.device:
        return list(self.parameters())[0].device
