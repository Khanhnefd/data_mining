from fastapi import APIRouter
from item import LogData, UserListenHistory
from function import (
    connect_mongo,
    insert_web_data,
    get_data_date,
    insert_listen_history,
    get_user_streaming_history,
)


from recommendation.predict_module import load_model, predict
from recommendation.data import RecSysDataset, Voc
from recommendation.data_util import collate_fn
from recommendation.util import get_track_ids, get_tracks_feature_data, load_file


from datetime import datetime
from typing import List
from torch.utils.data import DataLoader
from model.config import ModelConfig as Config
from random import sample
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)


router = APIRouter()

db_name = "spotify_logging"
db_user_history = "mev_user_history"
mongo_client = connect_mongo()

# we have: track_id <-> item_index (0, 1, 2, 3, ...)  ==> train model by item_index
list_track_id = get_track_ids()  # track id is 'adawhdwjdwd12831', 'jadwjdkawdk2891312'
feature_data = get_tracks_feature_data()


train = load_file("data/train.pkl")
# data train bao gồm: train_input_seq và train_input_label
# ex: [[214834865], [214834865, 214820225]]  -   [214820225, 214706441]
test = load_file("data/test.pkl")
voc = Voc("VocabularyItem")
voc.addSenquence([train[1]] + [test[1]])


@router.post("/recommendation")
async def recommendation_tracks(
    previous_track_id: List[str], number_recommend: int = 3
) -> list:
    model = load_model(
        checkpoint_path="model/latest_checkpoint_4.pt", voc=voc, device_id=0
    )
    result = []
    logging.info(f"previous track : {previous_track_id}")
    try:
        previous_item_id = [[list_track_id.index(id) for id in previous_track_id]]
        logging.info(f"previous_item_id: {previous_item_id}")

        input_data = RecSysDataset([previous_item_id, [0]])
        data_loader = DataLoader(
            input_data,
            batch_size=Config.batch_size_predict,
            shuffle=False,
            collate_fn=collate_fn,
        )

        predict_index_list = predict(
            loader=data_loader,
            model=model,
            topk=number_recommend,
            total_track_number=len(list_track_id),
            device_id=Config.device,
        )

        result = [list_track_id[index] for index in predict_index_list]
    except:
        logging.info("exception in /recommendation API")
        result = sample(list_track_id, number_recommend)

    logging.info(f"Result recommendation : {result}")
    return result


@router.post("/log-listen-history")
async def log_listen_history(listen_history: List[UserListenHistory]) -> list:
    db_cursor = mongo_client.get_database(db_user_history)
    result = await insert_listen_history(db=db_cursor, data=listen_history)
    return result


@router.post("/log-web-data")
async def log_web_data(log_data: LogData) -> dict:
    db_cursor = mongo_client.get_database(db_name)

    timestamp = log_data.timestamp
    data_date = await get_data_date(timestamp)
    collection_name = f"{data_date.month}_{data_date.year}"
    collection = db_cursor[collection_name]

    inserted_id = await insert_web_data(
        collection=collection, insert_data=log_data.dict()
    )

    response = {"collection_name": collection_name, "inserted_id": str(inserted_id)}
    return response


@router.get("/user-listen-history")
async def get_listen_history(
    user_id: str,
    limit: int = 20,
):
    db_cursor = mongo_client.get_database(db_user_history)

    result = await get_user_streaming_history(
        db=db_cursor, user_id=user_id, limit=limit
    )

    return result


@router.get("/payment-for-artist")
async def payment_artists(
    year: int = 2023,
    month: int = 10,
):
    db_cursor = mongo_client.get_database(db_name)

    collection_name = f"{month}_{year}"
    collection = db_cursor[collection_name]

    result = list(
        collection.aggregate(
            [
                {
                    "$match": {
                        "$expr": {
                            "$and": [
                                {"$eq": [{"$month": "$timestamp"}, 10]},
                                {"$eq": [{"$year": "$timestamp"}, 2023]},
                            ]
                        }
                    }
                },
                {"$unwind": {"path": "$data", "preserveNullAndEmptyArrays": False}},
                {
                    "$group": {
                        "_id": "$data.artistId",
                        "listen_total": {"$sum": "$data.listen"},
                    }
                },
            ]
        )
    )

    return result


@router.get("/artist-listen-statistic")
# get statistic of artist by specific date
async def get_artist_statistics(
    artist_id: str, year: int = 2023, month: int = 10, day: int = None
):
    db_cursor = mongo_client.get_database(db_name)

    collection_name = f"{month}_{year}"
    collection = db_cursor[collection_name]

    match_agg = {}

    # day None ==> get full month
    if day is not None:
        match_agg = {
            "$match": {
                "$expr": {
                    "$and": [
                        {"$eq": [{"$dayOfMonth": "$timestamp"}, day]},
                        {"$eq": [{"$month": "$timestamp"}, month]},
                        {"$eq": [{"$year": "$timestamp"}, year]},
                    ]
                }
            }
        }
    else:
        match_agg = {
            "$match": {
                "$expr": {
                    "$and": [
                        {"$eq": [{"$month": "$timestamp"}, month]},
                        {"$eq": [{"$year": "$timestamp"}, year]},
                    ]
                }
            }
        }
    result = list(
        collection.aggregate(
            [
                match_agg,
                {
                    "$project": {
                        "data": {
                            "$filter": {
                                "input": "$data",
                                "as": "d",
                                "cond": {"$eq": ["$$d.artistId", artist_id]},
                            }
                        },
                        "date": 1,
                    }
                },
                {"$match": {"data.artistId": artist_id}},
                {"$unwind": {"path": "$data"}},
                {
                    "$group": {
                        "_id": "$data.trackId",
                        "listen_total": {"$sum": "$data.listen"},
                    }
                },
            ]
        )
    )

    return result


@router.get("/web-listen-statistic")
# get statistic of website from date to date
async def get_web_statistic(
    artist_id: str = None,  # artist_id = None ==> get all web
    year: int = 2023,
    month: int = 10,
):
    db_cursor = mongo_client.get_database(db_name)

    collection_name = f"{month}_{year}"
    collection = db_cursor[collection_name]

    filter_artist = {
        "$project": {
            "data": {
                "$filter": {
                    "input": "$data",
                    "as": "d",
                    "cond": {"$eq": ["$$d.artistId", artist_id]},
                }
            },
            "date": 1,
        },
    }
    filter_empty = {"$match": {"data.artistId": artist_id}}

    pipeline = []

    if artist_id is None:
        pipeline = [
            {"$unwind": {"path": "$data"}},
            {"$group": {"_id": "$date", "listen_total": {"$sum": "$data.listen"}}},
        ]
    else:
        pipeline = [
            filter_artist,
            filter_empty,
            {"$unwind": {"path": "$data"}},
            {"$group": {"_id": "$date", "listen_total": {"$sum": "$data.listen"}}},
        ]

    result = list(collection.aggregate(pipeline))

    return result
