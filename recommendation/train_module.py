import os
import time
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn
from recommendation.data import RecSysDataset, Voc
from recommendation.data_util import collate_fn, load_data, get_data_continue_train
from model.config import ModelConfig as Config
from model.model import RecModel
from recommendation.util import set_device, validate, load_file
from recommendation.predict_module import load_model

device = set_device(device_id=0)


def training(train_data, test_data, voc, train_type: str = "continue"):
    print("Loading data...")
    train_data, valid_data, test_data = load_data(
        train_set=train_data, test_set=test_data
    )
    train_data = RecSysDataset(train_data)
    valid_data = RecSysDataset(valid_data)
    test_data = RecSysDataset(test_data)
    train_loader = DataLoader(
        train_data, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_data, batch_size=Config.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=Config.batch_size, shuffle=False, collate_fn=collate_fn
    )
    print("Complete load data!")

    model = None

    if train_type == "test":
        model = load_model(
            checkpoint_path="model/latest_checkpoint_4.pt", voc=voc, device_id=0
        )

        recall, mrr = validate(test_loader, model)
        print(
            "Test: Recall@{}: {:.4f}, MRR@{}: {:.4f}".format(
                Config.topk, recall, Config.topk, mrr
            )
        )
        return model

    elif train_type == "train":
        n_items = voc.num_items
        print(f"n_items = {n_items}")
        model = RecModel(
            hidden_size=Config.hidden_size,
            n_items=n_items,
            embedding_dim=Config.embed_dim,
            num_features=8,
            n_layers=2,
            dropout=0.25,
        ).to(device)
        print("complete create model!")

    elif train_type == "continue":
        model = load_model(
            checkpoint_path="model/latest_checkpoint_4.pt", voc=voc, device_id=0
        )
        print("complete load model!")


    optimizer = optim.Adam(model.parameters(), Config.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=Config.lr_dc_step, gamma=Config.lr_dc)

    print("start training!")
    previous_loss = 0
    for epoch in tqdm(range(Config.epoch)):
        # train for one epoch
        current_loss = trainForEpoch(
            train_loader,
            model,
            optimizer,
            epoch,
            Config.epoch,
            criterion,
            log_aggr=100,
        )
        scheduler.step()
        recall, mrr = validate(valid_loader, model)
        print(
            "Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n".format(
                epoch, Config.topk, recall, Config.topk, mrr
            )
        )

        if epoch == 0 or current_loss < previous_loss:
            ckpt_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": current_loss,
            }

            torch.save(
                ckpt_dict,
                "model/latest_checkpoint_4.pt",
            )
            print(f"Save checkpoint at epoch {epoch}")

        print(f"Epoch {epoch} has loss {current_loss}")

        previous_loss = current_loss

    return model


def trainForEpoch(
    train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=100
):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (seq, target, lens) in enumerate(train_loader):
        # print(f"batch {i}")

        seq = seq.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        outputs = model(seq, lens)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        if i % log_aggr == 0:
            print(
                "[TRAIN] epoch %d/%d  observation %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)"
                % (
                    epoch + 1,
                    num_epochs,
                    i,
                    len(train_loader),
                    loss_val,
                    sum_epoch_loss / (i + 1),
                    len(seq) / (time.time() - start),
                )
            )

        start = time.time()
    return sum_epoch_loss / (i + 1)


if __name__ == "__main__":
    train = load_file("data/train.pkl")
    test = load_file("data/test.pkl")
    voc = Voc("VocabularyItem")
    voc.addSenquence([train[1]] + [test[1]])

    print('sequence of itemIds: ', train[0][6])
    print('converted indices: ', voc._seqItem2seqIndex(train[0][6]))

    continue_train, continue_test = get_data_continue_train()

    train_x_index = [voc._seqItem2seqIndex(seq) for seq in continue_train[0]]
    test_x_index = [voc._seqItem2seqIndex(seq) for seq in continue_test[0]]
    train_y_index = voc._seqItem2seqIndex(train[1])
    test_y_index = voc._seqItem2seqIndex(test[1])
    train_index = (train_x_index, train_y_index)
    test_index = (test_x_index, test_y_index)

    # continue_train_data = None
    # continue_test_data = None

    training(train_data=train_index, test_data=test_index, voc=voc, train_type="continue")
