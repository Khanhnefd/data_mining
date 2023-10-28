import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from datetime import datetime, timedelta, date
import time

# db.pizzas.update_one(
#     {"_id": 1},
#     {
#         "$set": {"type": "kem chesese", "price": 10},
#         "$currentDate": {"lastModified": True}
#     }
# )

# collection.update(
#     {"timestamp": update_time},
#     {
#         "$push": {
#             "logging": {
#                 "id": 2 ,
#                 "data": update_data['data']
#             }
#         },
#         "$inc": {
#             "number_log_data": 1,
#             "total_listen": sum([i['listen'] for i in update_data['data']])
#         },
#     },
#     multi=True
# )

from dotenv import load_dotenv
import os

load_dotenv()

conn_str = os.getenv("conn_str")

def connect_mongo(db_name: str):
    # conn_str = "mongodb+srv://khanh:uZ3t2sbNaCiNlp5H@firstcluster.rccmvcx.mongodb.net/"
    try:
        client = MongoClient(conn_str)
        db = client.get_database(db_name)
        print("connect mongodb success")
        # print(list(db.list_collections(filter={"name": "8_2023"})))
        # print(db["8_2023"])

        return db
    except:
        print("connecting error")


def create_mongo_collection(db: Database, collection_name: str):
    db.create_collection(
        collection_name,
        timeseries={
            "timeField": "timestamp",
            "metaField": "date",
            "granularity": "minutes",
        }
        # expireAfterSeconds = 90000
    )


def find_document(collection: Collection, timestamp):
    x = collection.find({"timestamp": timestamp}, {})

    if list(x):
        return True
    else:
        return False


# datetime(year, month, day, hour, minute, second, microsecond)
time_test = datetime(2022, 12, 28, 23, 55, 0, 0)
time_test2 = datetime(2022, 12, 28, 23, 45, 0, 0)
time_test3 = datetime(2022, 12, 28, 23, 57, 0, 0)

time_today = datetime.now()


data = {
    "timestamp": "2023-04-09:12:30:00",
    "data": [
        {"track_id": "2372737dajgwdawjd", "listen": 777},
        {"track_id": "00007dajgwdawjd", "listen": 777},
    ],
}


def insert_mongo(collection: Collection, insert_data):

    insert_time = datetime.strptime(insert_data["timestamp"], f"%Y-%d-%m:%H:%M:%S")
    # print(insert_time.hour)
    # print(type(insert_time.hour))

    result = collection.insert_one(
        {
            "timestamp": insert_time,
            "date": insert_time.strftime(f"%Y-%d-%m"),
            "data": insert_data["data"],
            "total_listen": sum([i["listen"] for i in insert_data["data"]]),
        }
    )

    # # result = collection.insert_many(
    # #     [
    # #         # {
    # #         # "timestamp": time_test3,
    # #         # "logging": insert_data,
    # #         # "data": 2
    # #         # },
    # #         # {
    # #         # "timestamp": time_test,
    # #         # "logging": insert_data,
    # #         # "data": 1
    # #         # },
    # #         # {
    # #         # "timestamp": time_test2,
    # #         # "logging": insert_data,
    # #         # "data": 1
    # #         # },
    # #         # {
    # #         # "timestamp": time_test,
    # #         # "logging": insert_data,
    # #         # "data": 2
    # #         # },
    # #         # {
    # #         # "timestamp": time_test2,
    # #         # "logging": insert_data,
    # #         # "data": 2
    # #         # },
    # #     ]
    # # )

    print(result.inserted_id)


if __name__ == "__main__":
    db_name = "spotify_logging"
    db = connect_mongo(db_name)

    # create_mongo_collection(db, "8_2023")
    # create_mongo_collection(db, "9_2023")
    create_mongo_collection(db, "10_2023")
    # create_mongo_collection(db, "11_2023")
    # create_mongo_collection(db, "12_2023")
    # create_mongo_collection(db, "1_2024")

    # time.sleep(5)

    # insert_mongo(db["8_2023"], insert_data=data)

    # find_time = (time_today - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)

    # print(find_time)

    # print(find_document(db["8_2023"], find_time))

    # x = db["8_2023"].find({"timestamp": find_time}, {})

    # print(list(x))
