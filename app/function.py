import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from datetime import datetime, timedelta, date
from typing import List
import time
import logging
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")


async def connect_mongo() -> MongoClient:
    conn_str = f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@firstcluster.rccmvcx.mongodb.net/"
    try:
        client = MongoClient(conn_str)
        # db = client.get_database(db_name)
        # print("connect mongodb success")
        return client
    except:
        print("connecting error")


async def get_data_date(timestamp: str) -> datetime:
    datetime_timestamp = datetime.strptime(timestamp, f"%Y-%d-%m:%H:%M:%S")

    return datetime_timestamp


async def find_document(collection: Collection, timestamp):
    x = collection.find({"timestamp": timestamp}, {})

    if list(x):
        return True
    else:
        return False


async def insert_web_data(collection: Collection, insert_data: dict):

    insert_time = datetime.strptime(insert_data["timestamp"], f"%Y-%d-%m:%H:%M:%S")

    result = collection.insert_one(
        {
            "timestamp": insert_time,
            "date": insert_time.strftime(f"%Y-%d-%m"),
            "data": insert_data["data"],
            "total_listen": sum([i["listen"] for i in insert_data["data"]]),
        }
    )

    return result.inserted_id


async def insert_listen_history(db, data: List):
    result = []
    for d in data:
        listen_history = d.dict()
        collection_name = listen_history["userId"]
        history = listen_history["history"]
        for h in history:
            h.update(
                {"timestamp": datetime.strptime(h["timestamp"], f"%Y-%d-%m:%H:%M:%S")}
            )

        collection = db[collection_name]
        collection.insert_many(history)

        result.append(
            {"userId": listen_history["userId"], "listen_number": len(history)}
        )

    return result


async def get_user_streaming_history(db, user_id, limit):
    collection = db[user_id]
    result = list(
        collection.aggregate(
            [{"$sort": {"timestamp": -1}}, {"$limit": limit}, {"$project": {"_id": 0}}]
        )
    )
    return result
