import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

COLLECTION_NAME = "aqi_features"

_client = None


def _get_collection():
    global _client

    if _client is None:
        _client = MongoClient(os.getenv("MONGODB_URI"))

    db = _client[os.getenv("MONGODB_DB_NAME", "aqi_predictor")]

    return db[COLLECTION_NAME]


def get_collection():
    return _get_collection()


def insert_feature_row(row: dict) -> None:
    collection = _get_collection()
    collection.insert_one(dict(row))