# _*_ coding: utf-8 _*_
import pymongo
from settings import MONGO_DB, MONGO_PASSWORD, MONGO_URI, MONGO_USERNAME

uri = pymongo.MongoClient(MONGO_URI)
if MONGO_USERNAME != '' and MONGO_PASSWORD != '':
    uri[MONGO_DB].authenticate(name=MONGO_USERNAME, password=MONGO_PASSWORD)
db = uri[MONGO_DB]
