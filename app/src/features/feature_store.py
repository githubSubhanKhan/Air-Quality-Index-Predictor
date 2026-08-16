import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

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