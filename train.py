import pandas
import torch

from model.narre import NarreModel, NarreConfig
from utils.data_reader import get_train_dev_test_data, get_max_item_id, get_max_user_id
from utils.train_helper import train_model
from utils.word2vec_helper import load_embedding_weights, PAD_WORD_ID, WORD_EMBEDDING_SIZE

train_data, dev_data, test_data = get_train_dev_test_data()
know_data = pandas.concat([train_data, dev_data])

item_count = get_max_item_id() + 2
user_count = get_max_user_id() + 1
config = NarreConfig(
    num_epochs=100,
    batch_size=16,
    learning_rate=1e-3,
    l2_regularization=1e-3,
    learning_rate_decay=0.99,
    device="cuda:0" if torch.cuda.is_available() else "cpu",

    pad_word_id=PAD_WORD_ID,
    pad_item_id=item_count - 1,
    pad_user_id=user_count - 1,
    item_count=item_count,
    user_count=user_count,
    review_length=200,
    review_count=25,
    word_dim=WORD_EMBEDDING_SIZE,
    id_dim=32,
    kernel_width=5,
    kernel_deep=100
)

model = NarreModel(config, load_embedding_weights())
train_model(model, train_data, dev_data)
