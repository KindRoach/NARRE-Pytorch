from dataclasses import dataclass

import torch
import torch.nn.functional as F

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
    id_dim: int = 32

    kernel_width: int = 5
    kernel_deep: int = 100


class ReviewEncoder(torch.nn.Module):
    def __init__(self, config: NarreConfig, id_count: int):
        super().__init__()
        self.config = config

        self.id_embedding = torch.nn.Embedding(id_count, config.id_dim)

        self.conv = torch.nn.Conv1d(
            in_channels=config.word_dim,
            out_channels=config.kernel_deep,
            kernel_size=config.kernel_width,
            stride=1)
        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=config.review_length - config.kernel_width + 1,
            stride=1)

        self.att_review = torch.nn.Linear(config.review_count * config.kernel_deep, config.id_dim)
        self.att_id = torch.nn.Linear(config.id_dim, config.id_dim, bias=False)
        self.att_layer = torch.nn.Linear(config.id_dim, 1)

        self.top_linear = torch.nn.Linear(config.kernel_deep, config.id_dim)

    def forward(self, review, object_id):
        """
        Input Size:
            (Batch Size, Review Count, Review Length, Word Dim)
            (Batch Size, Review Count)

        Output Size:
            (Batch Size, Id Dim)
        """

        batch_size = review.view(0)
        review_in_one = review.view(batch_size, -1, self.config.word_dim)
        review_conv = F.relu(self.conv(review_in_one))
        review_conv = self.max_pool(review_conv)
        review_conv = torch.flatten(review_conv, dims=1)
        review_in_many = review_conv.view(batch_size, self.config.review_count, -1)

        id_emb = self.id_embedding(object_id)
        review_att = self.att_review(review_in_many)
        id_att = self.att_id(id_emb)
        att_weight = self.att_layer(F.relu(review_att + id_att))
        att_weight = F.softmax(att_weight)

        att_out = (att_weight * review_in_many).sum(1)
        att_out = self.top_linear(att_out)
        review_out = torch.stack([id_emb, att_out], dim=1)

        return review_out


class NarreModel(BaseModel):
    def __init__(self, config: NarreConfig, word_embedding_weight):
        super(NarreModel, self).__init__(config)
        self.config = config

        self.word_embedding = torch.nn.Embedding.from_pretrained(word_embedding_weight)
        self.word_embedding.weight.requires_grad = False

        self.user_review_layer = ReviewEncoder(config, config.user_count)
        self.item_review_layer = ReviewEncoder(config, config.item_count)

        self.predict_linear = torch.nn.Linear(config.id_dim * 2, 1)

    def forward(self, user_review, item_id, item_review, user_id):
        """
        Input Size:
            (Batch Size, Review Count, Review Length, Word Dim)
            (Batch Size, Review Count)
            (Batch Size, Review Count, Review Length, Word Dim)
            (Batch Size, Review Count)

        Output Size:
            (Batch Size, 1)
        """

        user_review = self.word_embedding(user_review)
        user_review = self.user_review_layer(user_review, item_id)

        item_review = self.word_embedding(item_review)
        item_review = self.user_review_layer(item_review, user_id)

        predict = self.predict_linear(user_review * item_review)
        return predict
