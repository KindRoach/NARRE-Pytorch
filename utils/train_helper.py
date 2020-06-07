import math
import time

import torch
from pandas import DataFrame
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from model.base_model import BaseModel, BaseConfig
from utils.data_reader import get_review_dict
from utils.data_set import NarreDataset
from utils.log_hepler import logger, add_log_file, remove_log_file
from utils.path_helper import ROOT_DIR


def save_model(model: torch.nn.Module, train_time: time.struct_time):
    path = "model/checkpoints/%s_%s.pt" % (
        model.__class__.__name__, time.strftime("%Y%m%d%H%M%S", train_time)
    )
    path = ROOT_DIR.joinpath(path)
    torch.save(model, path)
    logger.info(f"model saved: {path}")


def load_model(path: str):
    path = ROOT_DIR.joinpath(path)
    # load model to cpu as default.
    model = torch.load(path, map_location=torch.device('cpu'))
    return model


def eval_model(model, data_iter, loss):
    model.eval()
    model_name = model.__class__.__name__
    config: BaseConfig = model.config
    logger.debug("Evaluating %s..." % model_name)

    with torch.no_grad():
        predicts = []
        ratings = []
        for batch_id, iter_i in enumerate(data_iter):
            user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review, rating = iter_i

            user_review = user_review.to(config.device)
            user_id = user_id.to(config.device)
            item_id_per_review = item_id_per_review.to(config.device)

            item_review = item_review.to(config.device)
            item_id = item_id.to(config.device)
            user_id_per_review = user_id_per_review.to(config.device)

            rating = rating.to(config.device)

            predict = model(user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review)
            predicts.append(predict)
            ratings.append(rating)

        predicts = torch.cat(predicts)
        ratings = torch.cat(ratings)
        return loss(predicts, ratings).item()


def train_model(model: BaseModel, train_data: DataFrame, dev_data: DataFrame, is_save_model: bool = True):
    model_name = model.__class__.__name__
    train_time = time.localtime()
    add_log_file(logger, "log/%s_%s.log" % (model_name, time.strftime("%Y%m%d%H%M%S", train_time)))
    logger.info("Training %s..." % model_name)

    config: BaseConfig = model.config
    logger.info(config.__dict__)
    model.to(config.device)

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_s = lr_scheduler.ExponentialLR(opt, gamma=config.learning_rate_decay)
    loss = torch.nn.MSELoss()

    last_progress = 0.
    min_loss = float("inf")

    pin_memory = config.device not in ["cpu", "CPU"]
    review_by_user, review_by_item = get_review_dict("train")
    dataset = NarreDataset(train_data, review_by_user, review_by_item, config)
    train_data_iter = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=pin_memory)
    dataset = NarreDataset(dev_data, review_by_user, review_by_item, config)
    dev_data_iter = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=pin_memory)

    batches_num = math.ceil(len(train_data) / config.batch_size)
    while model.current_epoch < config.num_epochs:

        model.train()

        for batch_id, iter_i in enumerate(train_data_iter):
            user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review, rating = iter_i

            user_review = user_review.to(config.device)
            user_id = user_id.to(config.device)
            item_id_per_review = item_id_per_review.to(config.device)

            item_review = item_review.to(config.device)
            item_id = item_id.to(config.device)
            user_id_per_review = user_id_per_review.to(config.device)

            rating = rating.to(config.device)

            opt.zero_grad()
            predict = model(user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review)
            li = loss(predict, rating)
            li.backward()
            opt.step()

            # log progress
            current_batches = model.current_epoch * batches_num + batch_id + 1
            total_batches = config.num_epochs * batches_num
            progress = current_batches / total_batches
            if progress - last_progress > 0.001:
                logger.debug("epoch %d, batch %d, loss: %f (%.2f%%)" %
                             (model.current_epoch, batch_id, li.item(), 100 * progress))
                last_progress = progress

        # complete one epoch
        train_loss = eval_model(model, train_data_iter, loss)
        dev_loss = eval_model(model, dev_data_iter, loss)
        logger.info("Epoch %d complete. Total loss(train/dev)=%f/%f"
                    % (model.current_epoch, train_loss, dev_loss))

        # save best model
        if train_loss < min_loss:
            min_loss = train_loss
            logger.info(f"Get min loss: {train_loss}")
            if is_save_model:
                save_model(model, train_time)

        lr_s.step(model.current_epoch)
        model.current_epoch += 1

    logger.info("%s trained!" % model_name)
    remove_log_file(logger)
    return min_loss
