from pymongo import MongoClient
import json

# Replace with your MongoDB Atlas connection string
MONGO_URI = "mongodb+srv://venkatasaiancha:Venkat%4024@cluster1.xjjnc.mongodb.net/myFirstDatabase?tlsAllowInvalidCertificates=true"

def connect_to_mongodb():
    try:
        client = MongoClient(MONGO_URI)
        print("Connected to MongoDB Atlas!")
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None



def get_all_databases(client):
    """Retrieve all database names from MongoDB"""
    try:
        db_list = client.list_database_names() 
        print(" Available Databases:", db_list)
        return db_list
    except Exception as e:
        print(f" Error retrieving databases: {e}")
        return None


def get_all_collections(client, databases):
    all_collections = {}
    for db_name in databases:
        try:
            db = client[db_name]
            collections = db.list_collection_names()
            all_collections[db_name] = collections
        except Exception as e:
            print(f"Error fetching collections for database {db_name}: {e}")
    return all_collections

def infer_collection_schema(collection, sample_size=5):
    try:
        sample_documents = list(collection.find().limit(sample_size))

       
        schema = {}
        for doc in sample_documents:
            for key, value in doc.items():
                if key not in schema:
                    schema[key] = type(value).__name__
        return schema
    except Exception as e:
        print(f"Error inferring schema for collection {collection.name}: {e}")
        return None

def get_all_schemas(client, all_collections):
    all_schemas = {}
    for db_name, collections in all_collections.items():
        db = client[db_name]
        schemas = {}
        for collection_name in collections:
            collection = db[collection_name]
            schema = infer_collection_schema(collection)
            schemas[collection_name] = schema
        all_schemas[db_name] = schemas
    return all_schemas

client = connect_to_mongodb()
if client:
    databases = get_all_databases(client)
    all_collections = get_all_collections(client, databases)
    print(all_collections)
    all_schemas = get_all_schemas(client, all_collections)
    print(" ")
    print(all_schemas)