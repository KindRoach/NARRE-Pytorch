from typing import Set, List

import numpy as np
import pandas
import torch
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors

from utils.log_hepler import logger
from utils.path_helper import ROOT_DIR

PAD_WORD = "<pad>"
# Change this value according to the word embedding you use.
# Check code in get_word_vec()
PAD_WORD_ID = 3000000
WORD_EMBEDDING_SIZE = 300


def review2wid(review: str, word_vec: Word2VecKeyedVectors, max_length: int) -> List[int]:
    """
    1. Convert words in review to word idx, which is from pre-trained word embedding model.
    2. Pad or shorten review to max length.
    """

    wids = []
    pad_index = word_vec.vocab[PAD_WORD].index
    for word in review.split():
        if word in word_vec:
            wid = word_vec.vocab[word].index
        else:
            # PAD_WORD also used as UNK_WORD
            wid = pad_index
        wids.append(wid)

    # Pad list to max length.
    if len(wids) < max_length:
        wids += [pad_index] * (max_length - len(wids))
    else:
        wids = wids[:max_length]
    return wids


def get_word_vec(path='data/GoogleNews-vectors-negative300.bin'):
    """
    Read pre-trained word embedding model, and add "<pad>" to it with zero weight.
    """

    logger.info("loading word2vec model...")
    path = ROOT_DIR.joinpath(path)
    word_vec = KeyedVectors.load_word2vec_format(path, binary=True)

    if PAD_WORD not in word_vec:
        word_vec.add([PAD_WORD], np.zeros([1, 300]))
        logger.info(f"Add PAD_WORD to word embedding.")

    assert PAD_WORD_ID == word_vec.vocab[PAD_WORD].index, \
        f"PAD_WORD_ID should be {word_vec.vocab[PAD_WORD].index} but not {PAD_WORD_ID}."

    logger.info("word2vec model loaded.")
    return word_vec


def save_embedding_weights(word_vec, out_path="data/embedding_weight.pt"):
    """
    Save the weights of pre-trained word embedding model to file.
    Thus we don't need to load it when train our model.
    This helps to save RAM and model init time.
    """

    weight = torch.Tensor(word_vec.vectors)
    torch.save(weight, ROOT_DIR.joinpath(out_path))
    logger.info("Word embedding weight saved.")


def load_embedding_weights(path="data/embedding_weight.pt"):
    return torch.load(path)


# Find the unknowns words in review text.
# This step is not necessary for model train.
if __name__ == "__main__":
    df = pandas.read_json(ROOT_DIR.joinpath("data/reviews.json"), lines=True)
    word_vec = get_word_vec()
    unknown_words: Set[str] = set()
    for review in df["review"]:
        for word in review.split():
            if word not in word_vec:
                unknown_words.add(word)

    logger.warning(f"{len(unknown_words)} unknown words!")
    with open(ROOT_DIR.joinpath("out/UNKs.txt"), "w", encoding="utf-8") as f:
        for word in unknown_words:
            f.write(f"{word}\n")
