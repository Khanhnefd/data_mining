from pydantic import BaseModel
from typing import List, Union


class Data(BaseModel):
    trackId: str
    artistId: str
    listen: int


class LogData(BaseModel):
    timestamp: str
    data: List[Data]


class ListenHistory(BaseModel):
    listenDuration: str
    trackId: str
    timestamp: str


class UserListenHistory(BaseModel):
    userId: str
    history: List[ListenHistory]
