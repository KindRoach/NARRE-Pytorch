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

    dropout: float = 0.5


class ReviewEncoder(torch.nn.Module):
    def __init__(self, config: NarreConfig, preference_id_count: int, quality_id_count: int):
        super().__init__()
        self.config = config

        self.preference_id_embedding = torch.nn.Embedding(preference_id_count, config.id_dim)
        self.quality_id_embedding = torch.nn.Embedding(quality_id_count, config.id_dim)

        self.conv = torch.nn.Conv1d(
            in_channels=config.word_dim,
            out_channels=config.kernel_deep,
            kernel_size=config.kernel_width,
            stride=1)
        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=config.review_length - config.kernel_width + 1,
            stride=1)

        self.att_review = torch.nn.Linear(config.kernel_deep, config.id_dim)
        self.att_id = torch.nn.Linear(config.id_dim, config.id_dim, bias=False)
        self.att_layer = torch.nn.Linear(config.id_dim, 1)

        self.top_linear = torch.nn.Linear(config.kernel_deep, config.id_dim)
        self.dropout = torch.nn.Dropout(self.config.dropout)

    def forward(self, review_emb, preference_id, quality_id):
        """
        Input Size:
            (Batch Size, Review Count, Review Length, Word Dim)
            (Batch Size, Review Count)
            (Batch Size, Review Count)

        Output Size:
            (Batch Size, Id Dim)
        """

        preference_id_emb = self.preference_id_embedding(preference_id).view(-1, self.config.id_dim)
        quality_id_emb = self.quality_id_embedding(quality_id)

        batch_size = review_emb.shape[0]
        review_in_one = review_emb.view(-1, self.config.review_length, self.config.word_dim)
        review_in_one = review_in_one.permute(0, 2, 1)
        review_conv = F.relu(self.conv(review_in_one))
        review_conv = self.max_pool(review_conv).view(-1, self.config.kernel_deep)
        review_in_many = review_conv.view(batch_size, self.config.review_count, -1)

        review_att = self.att_review(review_in_many)
        id_att = self.att_id(quality_id_emb)
        att_weight = self.att_layer(F.relu(review_att + id_att))
        att_weight = F.softmax(att_weight, dim=1)
        att_out = (att_weight * review_in_many).sum(1)

        feature = self.dropout(att_out)
        feature = self.top_linear(feature)
        feature = preference_id_emb + feature
        return feature


class LatentFactor(torch.nn.Module):
    def __init__(self, config: NarreConfig):
        super().__init__()
        self.linear = torch.nn.Linear(config.id_dim, 1)
        self.b_user = torch.nn.Parameter(torch.randn([config.user_count]), requires_grad=True)
        self.b_item = torch.nn.Parameter(torch.randn([config.item_count]), requires_grad=True)

    def forward(self, user_feature, user_id, item_feature, item_id):
        """
        Input Size:
            (Batch Size, Id Dim)
            (Batch Size, 1)
            (Batch Size, Id Dim)
            (Batch Size, 1)

        Output Size:
            (Batch Size, 1)
        """
        dot = user_feature * item_feature
        predict = self.linear(dot) + self.b_user[user_id] + self.b_item[item_id]
        return predict


class NarreModel(BaseModel):
    def __init__(self, config: NarreConfig, word_embedding_weight):
        super().__init__(config)
        self.config = config

        self.word_embedding = torch.nn.Embedding.from_pretrained(word_embedding_weight)
        self.word_embedding.weight.requires_grad = False

        self.user_review_layer = ReviewEncoder(config, config.user_count, config.item_count)
        self.item_review_layer = ReviewEncoder(config, config.item_count, config.user_count)

        self.predict_linear = LatentFactor(config)

    def forward(self, user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review):
        """
        Input Size:
            (Batch Size, Review Count, Review Length, Word Dim)
            (Batch Size, 1)
            (Batch Size, Review Count)
            (Batch Size, Review Count, Review Length, Word Dim)
            (Batch Size, 1)
            (Batch Size, Review Count)

        Output Size:
            (Batch Size, 1)
        """

        user_review_emb = self.word_embedding(user_review)
        user_feature = self.user_review_layer(user_review_emb, user_id, item_id_per_review)

        item_review_emb = self.word_embedding(item_review)
        item_feature = self.item_review_layer(item_review_emb, item_id, user_id_per_review)

        predict = self.predict_linear(user_feature, user_id, item_feature, item_id)
        return predict
