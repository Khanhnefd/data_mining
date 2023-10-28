import torch
import logging
import pandas as pd
import pickle


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
        "data/data_spotify_track_feature_processed.pkl"
    )
    feature_data = feature_data_filter.set_index("id").T.to_dict("list")

    return feature_data


def load_file(filename):
    with open(filename, "rb") as fn:
        data = pickle.load(fn)
    return data
