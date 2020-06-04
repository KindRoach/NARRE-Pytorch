from typing import Dict, List

import torch

from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader

from utils.data_reader import get_review_dict, get_train_dev_test_data
from utils.log_hepler import logger
from utils.word2vec_hepler import PAD_WORD_ID


class NarreDateset(Dataset):
    def __init__(self, data: DataFrame,
                 user_review: Dict[str, DataFrame],
                 item_review: Dict[str, DataFrame],
                 review_count: int, review_length: int):
        """
        Init a NarreDateset.
        :param data: original data. ["userID","itemID","review","rating"]
        :param user_review: the review grouped by userID
        :param item_review: the review grouped by itemID
        :param review_count: the max count of review per user/item
        :param review_length: the length of review
        """

        super().__init__()
        self.data = data
        self.user_review = user_review
        self.item_review = item_review
        self.review_count = review_count
        self.review_length = review_length

        logger.info("Loading dataset...")

        self.user_reviews = []
        self.item_reviews = []

        for user_id, item_id in zip(self.data["userID"], self.data["itemID"]):
            user_review = self.load_reviews(self.user_review, user_id, item_id)
            item_review = self.load_reviews(self.item_review, item_id, user_id)
            self.user_reviews.append(user_review)
            self.item_reviews.append(item_review)

        self.ratings = self.data["rating"].to_list()

        self.user_reviews = torch.LongTensor(self.user_reviews)
        self.item_reviews = torch.LongTensor(self.item_reviews)
        self.ratings = torch.Tensor(self.ratings).view(-1, 1)

        logger.info("Dataset loaded.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.ratings[idx]

    def load_reviews(self, review: Dict[str, DataFrame], query_id: str, exclude_id: str) -> List[List[int]]:
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

            output = [[1, 2, 3],
                      [2, 3, 4],
                      [PAD_WORD_ID, PAD_WORD_ID, PAD_WORD_ID]]
        """

        reviews = review[query_id]
        key = "userID" if "userID" in reviews.columns else "itemID"
        reviews = reviews["review"][reviews[key] != exclude_id].to_list()
        reviews = [r[:self.review_length] for r in reviews]

        if len(reviews) >= self.review_count:
            reviews = reviews[:self.review_count]
        else:
            reviews += [[PAD_WORD_ID] * self.review_length] * (self.review_count - len(reviews))
        return reviews


if __name__ == '__main__':
    train_data, _, _ = get_train_dev_test_data()
    review_by_user, review_by_item = get_review_dict()
    dataset = NarreDateset(train_data, review_by_user, review_by_item, 20, 200)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for user_review, item_review, rating in loader:
        print(user_review.shape, item_review.shape, rating.shape)
