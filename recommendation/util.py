import torch
import logging
import pandas as pd
import pickle
import torch.nn.functional as F
import numpy as np
from model.config import ModelConfig as Config


def set_device(device_id: int = 0) -> torch.device:
    """
    Check cuda is available or not
    """
    logging.info("Check cuda is available or not:")

    # Check cuda in device
    if torch.cuda.is_available():
        if device_id != -1:
            torch.cuda.set_per_process_memory_fraction(0.5, device_id)
            device = torch.device(f"cuda:{device_id}")
            logging.info(torch.cuda.get_device_name())
            logging.info("Cuda is available")
        else:
            device = torch.device("cpu")
            logging.info("Cuda is not used")
    else:
        device = torch.device("cpu")
        logging.info("Cuda is not available")
    return device


def get_track_ids() -> list:
    list_track_id = []
    with open("data/track_id.txt", "r") as f:
        for x in f:
            list_track_id.append(x.strip())

    return list_track_id


def get_tracks_feature_data() -> dict:
    feature_data_filter = pd.read_pickle(
        "data/data_track_feature_processed.pkl"
    )
    feature_data = feature_data_filter.set_index("id").T.to_dict("list")

    return feature_data


def load_file(filename):
    with open(filename, "rb") as fn:
        data = pickle.load(fn)
    return data

def get_recall(indices, targets):
    device = set_device(0)
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).to(device)
    hits = hits.double()
    if targets.size(0) == 0:
        return 0
    n_hits = torch.sum(hits)
    recall = n_hits / targets.size(0)
    return recall


def get_mrr(indices, targets):
    device = set_device(0)
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).to(device)
    hits = hits.double()
    if hits.sum() == 0:
      return 0
    argsort = []
    for i in np.arange(hits.shape[0]):
      index_col = torch.where(hits[i, :] == 1)[0]+1
      if index_col.shape[0] != 0:
        argsort.append(index_col.double())
    inv_argsort = [1/item for item in argsort]
    mrr = sum(inv_argsort)/hits.size(0)
    return mrr


def evaluate(logits, targets, k=20):
    _, indices = torch.topk(logits, k, -1)
    recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)
    return recall, mrr

def validate(valid_loader, model):
    device = set_device(0)
    model.eval()
    recalls = []
    mrrs = []
    with torch.no_grad():
        for seq, target, lens in valid_loader:
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim = 1)
            recall, mrr = evaluate(logits, target, k = Config.topk)
            recalls.append(recall)
            mrrs.append(mrr)
    print("recalls:", recalls)
    print("mrrs:", mrrs)

    mean_recall = torch.mean(torch.stack(recalls))
    mean_mrr = torch.mean(torch.stack(mrrs))
    return mean_recall, mean_mrr
