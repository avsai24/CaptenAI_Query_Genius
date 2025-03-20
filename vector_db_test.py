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
CHROMA_DB_PATH = "/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/vector_database"


def delete_folders_except_sqlite(parent_directory):
    
    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)

        if os.path.isdir(item_path):  
            shutil.rmtree(item_path)
            print(f"Deleted folder: {item_path}")
        elif item.endswith(".sqlite3"):  
            print(f"Keeping SQLite file: {item_path}")


def get_user_collection():
    
    if "username" not in st.session_state:
        logging.error("User is not logged in. Cannot create ChromaDB collection.")
        return None

    username = st.session_state["username"]
    collection_name = f"user_{username}"  
    logging.info(f"Using ChromaDB collection: {collection_name}")

    return Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)

def print_all_collections_in_vectordb():
    chroma_db = get_user_collection()
    available_collections = chroma_db._client.list_collections()
    print("Existing Collections:", available_collections)

def get_info_inside_collection():
    chroma_db = get_user_collection()
    documents= chroma_db._collection.get()
    pprint.pp(documents)

def is_new_user():
    chroma_db = get_user_collection()
    
    documents= chroma_db._collection.get()
    if not documents["metadatas"]:
        return True
    return False

def get_new_chat_id():
    chroma_db = get_user_collection()
    documents= chroma_db._collection.get(include=["metadatas"])
    
    chat_ids = [int(meta["chat_id"]) for meta in documents["metadatas"] if "chat_id" in meta]
    if not chat_ids:
        return 1
    max_chat_id = max(set(chat_ids))

    return max_chat_id+1

def get_chat_ids():
    chroma_db = get_user_collection()
    documents= chroma_db._collection.get(include=["metadatas"])
    
    chat_ids = {
        int(meta["chat_id"]) for meta in documents["metadatas"]
        if "chat_id" in meta and isinstance(meta["chat_id"], (int, str)) and str(meta["chat_id"]).strip().isdigit()
    }

    if not chat_ids:
        return None

    # Sort chat IDs in descending order
    chat_ids = sorted(chat_ids, reverse=True)

    # Remove the currently active chat ID (if it exists in session state)
    if "chat_id" in st.session_state:
        current_chat_id = st.session_state["chat_id"]
        chat_ids = [cid for cid in chat_ids if cid != current_chat_id]

    return chat_ids

def add_conversation_to_new_chat(user_input, model_response, final_response, sqlitedb_arr, mongodb_arr, sqlitedb_df,
                    st_sqlitedb_arr="None", st_mongodb_arr="None", st_sqlitedb_df="None", generated_chart="None"):

    logging.info("Entered add_conversation.")
    new_user = is_new_user()
    if new_user:
        chat_id = 1
    else:
        id = get_new_chat_id()
        if id:
            chat_id = id+1


    chroma_db = get_user_collection()
    if chroma_db is None:
        return  

    timestamp = int(time.time())  
    doc = Document(
        page_content=f"User: {user_input}\nModel: {model_response}",
        metadata={
            "chat_id": chat_id,
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

def add_conversation_to_existing_chat(user_input, model_response, final_response, sqlitedb_arr, mongodb_arr, sqlitedb_df,
                    st_sqlitedb_arr="None", st_mongodb_arr="None", st_sqlitedb_df="None", generated_chart="None"):

    chroma_db = get_user_collection()
    if chroma_db is None:
        return  
    chat_id = st.session_state.get("chat_id", 1)
    logging.info(f"chat_id = {chat_id}")
    timestamp = int(time.time())  
    doc = Document(
        page_content=f"User: {user_input}\nModel: {model_response}",
        metadata={
            "chat_id": chat_id,
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
    add = chroma_db.add_documents([doc])
    delete_folders_except_sqlite(CHROMA_DB_PATH)
    logging.info(f"{add}")
    logging.info("Conversation added successfully.")

def get_chats_by_id(chat_id):
    chroma_db = get_user_collection()
    documents = chroma_db._collection.get(include=["metadatas"])
    
    filtered_chats = [
        {   
            "chat_id": meta.get("chat_id"),
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
        }
        for meta in documents["metadatas"] if meta.get("chat_id") == chat_id  
    ]
    sorted_chats = sorted(filtered_chats, key=lambda x: x["timestamp"])
    return sorted_chats

def get_document_ids_by_chat_id(chat_id):
    chroma_db = get_user_collection()
    documents = chroma_db._collection.get(include=["metadatas"])

    doc_ids = chroma_db._collection.peek()["ids"]

    document_ids = [
        doc_id for doc_id, meta in zip(doc_ids, documents["metadatas"])
        if (meta.get("chat_id")) == chat_id  
    ]

    return document_ids

def delete_chat_by_id(chat_id):
    chroma_db = get_user_collection()
    ids = get_document_ids_by_chat_id(chat_id)
    if not ids:
        st.sidebar.write("no conversations to delete")
    else:
        chroma_db.delete(ids=ids)
        st.sidebar.write(f"completed deleting conversation of chat_id:{chat_id}")

def delete_collection():
    chroma_db = get_user_collection()
    chroma_db.delete_collection()
    logging.info("deleted collection successfully")




if __name__ == "__main__":
    
    # add_conversation_to_new_chat(user_input = username, model_response = "model response 3", final_response="final response 3",
    #                             sqlitedb_arr = "sqlite_db_arr 3", mongodb_arr = "mongo_db_arr 3", sqlitedb_df = "sqlitedb_df 3")

    # add_conversation_to_existing_chat(user_input = username, model_response = "model response5", final_response="final response 5",
    #                         sqlitedb_arr = "sqlite_db_arr 5", mongodb_arr = "mongo_db_arr 5", sqlitedb_df = "sqlitedb_df 5")

    # chat = get_document_ids_by_chat_id(4)
    # pprint.pp(chat)
    # delete_chat_by_id(4)
    print_all_collections_in_vectordb()