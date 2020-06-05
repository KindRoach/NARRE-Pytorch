from dataclasses import dataclass
from typing import Dict, List

import torch

from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader

from model.narre import NarreConfig
from utils.data_reader import get_review_dict, get_train_dev_test_data, get_max_user_id, get_max_item_id
from utils.log_hepler import logger
from utils.word2vec_helper import PAD_WORD_ID


class NarreDataset(Dataset):
    def __init__(self, data: DataFrame,
                 user_review: Dict[str, DataFrame],
                 item_review: Dict[str, DataFrame],
                 config: NarreConfig):
        """
        Init a NarreDateset.
        :param data: original data. ["userID","itemID","review","rating"]
        :param user_review: the review grouped by userID
        :param item_review: the review grouped by itemID
        :param config: the config of Narre model.
        """

        super().__init__()
        self.data = data
        self.user_review = user_review
        self.item_review = item_review
        self.config = config

        logger.info("Loading dataset...")

        self.user_reviews, self.item_ids = self.load_user_review_data()
        self.item_reviews, self.user_ids = self.load_item_review_data()

        ratings = self.data["rating"].to_list()
        self.ratings = torch.Tensor(ratings).view(-1, 1)

        logger.info("Dataset loaded.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> (
            torch.LongTensor,
            torch.LongTensor,
            torch.LongTensor,
            torch.LongTensor,
            torch.Tensor):

        return self.user_reviews[idx], \
               self.item_ids[idx], \
               self.item_reviews[idx], \
               self.user_ids[idx], \
               self.ratings[idx]

    def load_user_review_data(self) -> (torch.LongTensor, torch.LongTensor):
        user_reviews = []
        item_ids_per_review = []

        for user_id, item_id in zip(self.data["userID"], self.data["itemID"]):
            u_review, i_id = self.load_reviews(self.user_review, user_id, item_id)
            user_reviews.append(u_review)
            item_ids_per_review.append(i_id)

        return torch.LongTensor(user_reviews), torch.LongTensor(item_ids_per_review)

    def load_item_review_data(self) -> (torch.LongTensor, torch.LongTensor):
        item_reviews = []
        user_ids_per_review = []

        for user_id, item_id in zip(self.data["userID"], self.data["itemID"]):
            i_review, u_id = self.load_reviews(self.item_review, item_id, user_id)
            item_reviews.append(i_review)
            user_ids_per_review.append(u_id)

        return torch.LongTensor(item_reviews), torch.LongTensor(user_ids_per_review)

    def load_reviews(self, review: Dict[str, DataFrame], query_id: str, exclude_id: str) \
            -> (List[List[int]], List[int]):
        """
        1. Load review from review dict by userID/itemID
        2. Exclude unknown review by itemID/userID.
        3. Pad or shorten review text to max_count

        E.g. get all reviews written by user1 except itemA
             when we predict the rating of itemA marked by user1.

            DataFrame for user1:

                | itemID | review |
                | itemA  | 0,1,2  |
                | itemB  | 1,2,3  |
                | itemC  | 2,3,4  |

            query_id: user1
            exclude_id: itemA
            max_count: 3
            review_length: 3

            output =(
              [[1, 2, 3],
              [2, 3, 4],
              [PAD_WORD_ID, PAD_WORD_ID, PAD_WORD_ID]],

              [itemB,itemC,PAD_ITEM_ID]
            )

        """

        config = self.config
        reviews = review[query_id]
        if "userID" in reviews.columns:
            id_name = "userID"
            pad_id = config.pad_user_id
        else:
            id_name = "itemID"
            pad_id = config.pad_item_id

        ids = reviews[id_name][reviews[id_name] != exclude_id].to_list()
        reviews = reviews["review"][reviews[id_name] != exclude_id].to_list()

        # shorten review length
        reviews = [r[:config.review_length] for r in reviews]
        # pad review length
        reviews = [r + [config.pad_word_id] * (config.review_length - len(r)) for r in reviews]

        # shorten review count
        reviews = reviews[:config.review_count]
        ids = ids[:config.review_count]
        # pad review count
        pad_length = config.review_count - len(reviews)
        pad_review = [config.pad_word_id] * config.review_length
        reviews += [pad_review] * pad_length
        ids += [pad_id] * pad_length

        return reviews, ids


if __name__ == '__main__':
    train_data, _, _ = get_train_dev_test_data()
    review_by_user, review_by_item = get_review_dict("train")

    config = NarreConfig(
        review_length=50,
        review_count=20,
        pad_word_id=PAD_WORD_ID,
        pad_item_id=get_max_item_id() + 1,
        pad_user_id=get_max_user_id() + 1
    )

    dataset = NarreDataset(train_data, review_by_user, review_by_item, config)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for user_review, item_ids, item_review, user_ids, rating in loader:
        logger.info(
            f"{user_review.shape}, "
            f"{item_ids.shape}, "
            f"{item_review.shape}, "
            f"{user_ids.shape}, "
            f"{rating.shape}")
