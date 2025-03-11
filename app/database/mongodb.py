import os
from pymongo import MongoClient

def get_database():
    """
    Connects to MongoDB using environment variables for flexibility.
    Returns the database instance.
    """
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGO_DB_NAME", "default_db")

    client = MongoClient(mongo_uri)
    return client[db_name]
