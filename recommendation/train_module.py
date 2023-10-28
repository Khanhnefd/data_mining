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


def main():
    print("Loading data...")
    train_data, valid_data, test_data = load_data(
        train_set=train_index, test_set=test_index
    )
    train_data = RecSysDataset(train_data)
    valid_data = RecSysDataset(valid_data)
    test_data = RecSysDataset(test_data)
    train_loader = DataLoader(
        train_data, batch_size=args["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_data, batch_size=args["batch_size"], shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=args["batch_size"], shuffle=False, collate_fn=collate_fn
    )
    print("Complete load data!")
    n_items = voc.num_items
    model = NARM(
        hidden_size=args["hidden_size"],
        n_items=n_items,
        embedding_dim=args["embed_dim"],
        num_features=8,
        n_layers=2,
        dropout=0.25,
    ).to(device)
    print("complete load model!")

    if args["test"] == "store_true":
        ckpt = torch.load(
            "/content/drive/MyDrive/Đồ án model recommendation/checkpoint_model/full_data_increst_test_set/latest_checkpoint_4.pt"
        )
        model.load_state_dict(ckpt["state_dict"])
        # recall, mrr = validate(test_loader, model)
        # print("Test: Recall@{}: {:.4f}, MRR@{}: {:.4f}".format(args['topk'], recall, args['topk'], mrr))
        return model

    optimizer = optim.Adam(model.parameters(), args["lr"])
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=args["lr_dc_step"], gamma=args["lr_dc"])

    print("start training!")
    previous_loss = 0
    for epoch in tqdm(range(args["epoch"])):
        # train for one epoch
        current_loss = trainForEpoch(
            train_loader,
            model,
            optimizer,
            epoch,
            args["epoch"],
            criterion,
            log_aggr=100,
        )
        scheduler.step()
        recall, mrr = validate(valid_loader, model)
        print(
            "Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n".format(
                epoch, args["topk"], recall, args["topk"], mrr
            )
        )

        # wandb.log({
        #     f"recall@{args['topk']}": recall,
        #     "mrr": mrr,
        #     "epoch": epoch + 1,
        # })

        if epoch == 0 or current_loss < previous_loss:
            # store best loss and save a model checkpoint
            ckpt_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": current_loss,
            }

            torch.save(
                ckpt_dict,
                "/content/drive/MyDrive/Đồ án model recommendation/checkpoint_model/full_data_increst_test_set/latest_checkpoint_4.pt",
            )
            # torch.save(ckpt_dict, '/content/drive/MyDrive/Đồ án model recommendation/checkpoint_model/full_data/latest_checkpoint_3.pt')

            # torch.save(ckpt_dict, '/content/drive/MyDrive/Đồ án model recommendation/checkpoint_model/latest_checkpoint_2.pt')

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

        iter_num = epoch * len(train_loader) + i + 1

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

            loss_total.append(loss_val)

            wandb.log({"loss_val": loss_val})

        start = time.time()

        # loss_total.append(sum_epoch_loss / (i + 1))

    return sum_epoch_loss / (i + 1)
