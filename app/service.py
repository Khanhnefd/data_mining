from fastapi import APIRouter
from item import LogData, UserListenHistory
from function import (
    connect_mongo,
    insert_web_data,
    get_data_date,
    insert_listen_history,
    get_user_streaming_history,
)
from datetime import datetime
from typing import List

router = APIRouter()

db_name = "spotify_logging"
db_user_history = "mev_user_history"


@router.post("/log-listen-history")
async def log_listen_history(listen_history: List[UserListenHistory]) -> list:
    mongo_client = await connect_mongo()
    db_cursor = mongo_client.get_database(db_user_history)

    result = await insert_listen_history(db=db_cursor, data=listen_history)

    return result


@router.post("/log-web-data")
async def log_web_data(log_data: LogData) -> dict:
    mongo_client = await connect_mongo()
    db_cursor = mongo_client.get_database(db_name)

    timestamp = log_data.timestamp
    data_date = await get_data_date(timestamp)
    collection_name = f"{data_date.month}_{data_date.year}"
    collection = db_cursor[collection_name]

    inserted_id = await insert_web_data(
        collection=collection, insert_data=log_data.dict()
    )

    mongo_client.close()

    response = {"collection_name": collection_name, "inserted_id": str(inserted_id)}
    return response


@router.get("/user-listen-history")
async def get_listen_history(
    user_id: str,
    limit: int = 20,
):
    mongo_client = await connect_mongo()
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
    mongo_client = await connect_mongo()
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
async def get_artist_statistics(
    artist_id: str, year: int = 2023, month: int = 10, day: int = None
):
    mongo_client = await connect_mongo()
    db_cursor = mongo_client.get_database(db_name)

    collection_name = f"{month}_{year}"
    collection = db_cursor[collection_name]

    match_agg = {}

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

