import torch
from torch.utils.data import DataLoader

from utils.data_reader import get_train_dev_test_data, get_review_dict
from utils.data_set import NarreDataset
from utils.log_hepler import logger
from utils.train_helper import load_model, eval_model

train_data, dev_data, test_data = get_train_dev_test_data()
model = load_model("model/checkpoints/NarreModel_20200606153827.pt")
model.config.device = "cuda:1"
model.to(model.config.device)
loss = torch.nn.MSELoss()

review_by_user, review_by_item = get_review_dict("test")
dataset = NarreDataset(test_data, review_by_user, review_by_item, model.config)
data_iter = DataLoader(dataset, batch_size=model.config.batch_size, shuffle=True)

logger.info(f"Loss on test dataset: {eval_model(model, data_iter, loss)}")
