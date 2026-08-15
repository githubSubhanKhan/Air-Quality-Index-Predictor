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

    collection = db[COLLECTION_NAME]

    collection.create_index(
        [
            ("city", 1),
            ("timestamp", 1),
        ],
        unique=True,
    )

    return collection


def insert_feature_row(row: dict) -> None:
    """
    Insert or update a feature row.
    Prevents duplicate city/timestamp records.
    """

    collection = _get_collection()

    collection.update_one(
        {
            "city": row["city"],
            "timestamp": row["timestamp"],
        },
        {
            "$set": dict(row),
        },
        upsert=True,
    )
