import json
import os

basedir = os.path.abspath(os.path.dirname(__file__))


class ModelConfig:
    config_path = "config.json"
    config = json.load(open(os.path.join(basedir, config_path), "r"))

    batch_size = config["batch_size"]
    batch_size_predict = config["batch_size_predict"]
    hidden_size = config["hidden_size"]
    embed_dim = config["embed_dim"]
    epoch = config["epoch"]
    lr = config["lr"]
    lr_dc = config["lr_dc"]
    lr_dc_step = config["lr_dc_step"]
    topk = config["topk"]
    valid_portion = config["valid_portion"]
    device = config["device"]
    num_features = config["num_features"]
    n_layers = config["n_layers"]
    dropout = config["dropout"]
    #  = config[""]
