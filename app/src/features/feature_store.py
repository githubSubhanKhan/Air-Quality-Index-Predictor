import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

load_dotenv()

COLLECTION_NAME = "aqi_features"

_client = None


def _get_client():
    global _client

    if _client is None:
        _client = MongoClient(os.getenv("MONGODB_URI"))

    return _client


def get_database():
    """The project database — shared by the feature store and model registry."""

    return _get_client()[os.getenv("MONGODB_DB_NAME", "aqi_predictor")]


def _get_collection():
    return get_database()[COLLECTION_NAME]


def get_collection():
    return _get_collection()


def insert_feature_row(row: dict) -> None:
    collection = _get_collection()

    try:
        collection.update_one(
        {
            "city": row["city"],
            "timestamp": row["timestamp"],
        },
        {"$set": row},
        upsert=True,
    )
        print("Inserted:", row["timestamp"])

    except DuplicateKeyError:
        print(
            f"Duplicate skipped: "
            f"{row['city']} - {row['timestamp']}"
        )