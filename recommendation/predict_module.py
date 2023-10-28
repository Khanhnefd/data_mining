import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from .util import set_device
from model.model import RecModel
from model.config import ModelConfig as Config
from .data import Voc
import logging


def load_model(checkpoint_path: str, voc: Voc, device_id: int = 0):
    checkpoint = torch.load(checkpoint_path)
    device = set_device(device_id=device_id)

    model = RecModel(
        hidden_size=Config.hidden_size,
        n_items=voc.num_items,
        embedding_dim=Config.embed_dim,
        num_features=Config.num_features,
        n_layers=Config.n_layers,
        dropout=Config.dropout,
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    return model


def predict(
    loader, model, topk=Config.topk, total_track_number: int = 120, device_id: int = 0
):
    model.eval()
    device = set_device(device_id=device_id)
    with torch.no_grad():
        for seq, _, lens in loader:
            seq = seq.to(device)
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim=1)
            logits = logits[:, :total_track_number]
            prob, indices = torch.topk(logits, topk, -1)
    logging.info(f"softmax prob : {prob}")

    return indices[0].tolist()
