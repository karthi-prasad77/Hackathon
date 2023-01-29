from model import Data
from pymongo import MongoClient

mongodb_uri = "mongodb://localhost:27017/"
client = MongoClient(mongodb_uri)

database = client['User']