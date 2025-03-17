from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import logging
import time
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pprint 
from chromadb import PersistentClient

logging.basicConfig(
    level=logging.INFO, 
    format="---------- %(levelname)s - %(message)s ----------", 
)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

CHROMA_DB_PATH = "/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/vector_database"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_user_chroma_db():
    
    username = "av"
    collection_name = f"user_{username}" 
    logging.info(f"Using ChromaDB collection: {collection_name}")

    return Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)

def add_conversation(user_input, model_response, final_response, sqlitedb_arr="None", mongodb_arr="None", sqlitedb_df="None"):
    
    logging.info("Adding conversation...")

    chroma_db = get_user_chroma_db()
    if chroma_db is None:
        return  

    timestamp = int(time.time())  
    doc = Document(
        page_content=f"User: {user_input}\nModel: {model_response}",
        metadata={
            "user_input": user_input,
            "model_response": model_response,
            "final_response": final_response,
            "sqlitedb_arr": sqlitedb_arr,
            "mongodb_arr": mongodb_arr,
            "sqlitedb_df": sqlitedb_df,
            "timestamp": timestamp
        }
    )
    add = chroma_db.add_documents([doc])
    logging.info("Conversation added successfully.")
    return add

def load_conversations():
    logging.info("Loading conversations...")
    
    chroma_db = get_user_chroma_db()
    if chroma_db is None:
        return []

    documents = chroma_db._collection.get(include=["metadatas"])
    if not documents["metadatas"]:
        logging.info("No conversations found.")
        return []

    conversations = sorted(documents["metadatas"], key=lambda x: x["timestamp"])
    return conversations

def delete_conversations():
    logging.info("Clearing all conversations...")

    chroma_db = get_user_chroma_db()
    if chroma_db is None:
        logging.warning("No collection found. Cannot clear conversations.")
        return

    document_ids = chroma_db._collection.get()["ids"]        
    if not document_ids:
        logging.warning("No conversations to delete.")
        return
        
    chroma_db.delete(ids=document_ids)  
    logging.info("All conversations deleted successfully.")

def delete_collection():
   
    logging.info("Deleting the entire ChromaDB collection...")

    chroma_db = get_user_chroma_db()
    if chroma_db is None:
        logging.warning("No collection found. Cannot delete.")
        return

    chroma_db.delete_collection()  
    logging.info("Collection deleted successfully.")

def print_collection_metadata():

    logging.info("Retrieving metadata from the collection...")

    chroma_db = get_user_chroma_db()
    if chroma_db is None:
        logging.warning("No collection found.")
        return

    documents = chroma_db.get(include=["metadatas"])
    
    if not documents["metadatas"]:
        logging.info("No metadata found in the collection.")
        return

    logging.info("Metadata in collection:")
    for index, meta in enumerate(documents["metadatas"]):
        print(f" **Document {index + 1} Metadata:**")
        for key, value in meta.items():
            print(f"   ➤ {key}: {value}")
        print("-" * 50)




def print_all_collections():
    
    client = PersistentClient(path=CHROMA_DB_PATH)
    collections = client.list_collections()

    if not collections:
        print("No collections found in ChromaDB.")
        return

    print("**Available Collections in ChromaDB:**")
    for collection in collections:
        print(f" {collection.name}")
    print("-" * 50)


### **Streamlit UI for Testing** ###
if __name__ == "__main__":
    
    # add_conversation(user_input="avsai1",model_response="avsai1",final_response="avsai1")
    load=load_conversations()
    pprint.pp(load)
    # delete_conversations()
    