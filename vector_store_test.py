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
import uuid
import os
import shutil

logging.basicConfig(
    level=logging.INFO, 
    format="---------- %(levelname)s - %(message)s ----------", 
)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DB_PATH = "/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/VECTOR_STORE"

def delete_folders_except_sqlite(parent_directory):
    
    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)

        if os.path.isdir(item_path):  
            shutil.rmtree(item_path)
            logging.info(f"Deleted folder: {item_path}")
        elif item.endswith(".sqlite3"):  
            logging.info(f"Keeping SQLite file")

def get_user_chroma_db():
    

    username = "avsai"
    collection_name = f"user_{username}"  
    logging.info(f"Using ChromaDB collection: {collection_name}")

    return Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)



def add_conversation(user_input, model_response, final_response, sqlitedb_arr, mongodb_arr, sqlitedb_df,
                    st_sqlitedb_arr="None", st_mongodb_arr="None", st_sqlitedb_df="None", generated_chart="None"):

    logging.info("Entered add_conversation.")

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
            "st_sqlitedb_arr": st_sqlitedb_arr,
            "st_mongodb_arr": st_mongodb_arr,
            "st_sqlitedb_df": st_sqlitedb_df,
            "generated_chart": generated_chart,
            "timestamp": timestamp
        }
    )
    chroma_db.add_documents([doc])
    delete_folders_except_sqlite(CHROMA_DB_PATH)
    logging.info("Conversation added successfully.")

def load_conversations():
    logging.info("Entered load_conversations.")
    
    chroma_db = get_user_chroma_db()
    if chroma_db is None:
        return []

    documents = chroma_db._collection.get(include=["metadatas"])
    if not documents["metadatas"]:
        logging.info("No conversations found for this user.")
        return []

    conversations = []
    for meta in documents["metadatas"]:
        conversations.append({
            "user_input": meta.get("user_input"),
            "model_response": meta.get("model_response"),
            "final_response": meta.get("final_response"),
            "sqlitedb_arr": meta.get("sqlitedb_arr"),
            "mongodb_arr": meta.get("mongodb_arr"),
            "sqlitedb_df": meta.get("sqlitedb_df"),
            "st_sqlitedb_arr": meta.get("st_sqlitedb_arr"),
            "st_mongodb_arr": meta.get("st_mongodb_arr"),
            "st_sqlitedb_df": meta.get("st_sqlitedb_df"),
            "generated_chart": meta.get("generated_chart"),
            "timestamp": meta.get("timestamp", 0)
        })

    return sorted(conversations, key=lambda x: x["timestamp"])

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
    delete_folders_except_sqlite(CHROMA_DB_PATH)





### **Streamlit UI for Testing** ###
if __name__ == "__main__":
    
    # add_conversation(user_input="avsai1",model_response="avsai1",final_response="avsai1",sqlitedb_arr="None", mongodb_arr="None", sqlitedb_df="None")
    load=load_conversations()
    pprint.pp(load)
    # delete_collection()
    