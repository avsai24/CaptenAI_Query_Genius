from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
from langchain_community.vectorstores.utils import filter_complex_metadata  # Ensure compatibility
import json
import streamlit as st


logging.basicConfig(
    level=logging.INFO, 
    format="---------- %(levelname)s - %(message)s ----------", 
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db = Chroma(collection_name="conversation_history", embedding_function=embeddings, persist_directory="/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/vector_db_files/chroma_db")

def add_conversation(user_input,model_response,final_response,sqlitedb_arr,mongodb_arr,sqlitedb_df,
                    st_sqlitedb_arr= "None",
                    st_mongodb_arr="None",
                    st_sqlitedb_df="None",
                    generated_chart="None"):
    logging.info("entered add conversation.")
    timestamp = int(time.time()) 
    doc = Document(
        page_content=f"User: {user_input}\nModel: {model_response}",
        metadata={"user_input": user_input,
                "model_response": model_response,
                "final_response": final_response,
                "sqlitedb_arr":sqlitedb_arr,
                "mongodb_arr":mongodb_arr,
                "sqlitedb_df":sqlitedb_df,
                "st_sqlitedb_arr":st_sqlitedb_arr,
                "st_mongodb_arr":st_mongodb_arr,
                "st_sqlitedb_df": st_sqlitedb_df,
                "generated_chart":generated_chart,
                "timestamp": timestamp}
    )
    chroma_db.add_documents([doc])
   

def load_conversations():
    logging.info("entered load conversations.")
    documents = chroma_db._collection.get(include=["metadatas"]) 
    if not documents["metadatas"]:
        logging.info("No conversations found in the database.")
        return []
    conversations = []
    for meta in documents["metadatas"]:
            conversations.append({
                "user_input": meta.get("user_input"),
                "model_response": meta.get("model_response"),
                "final_response": meta.get("final_response"),
                "sqlitedb_arr":meta.get("sqlitedb_arr"),
                "mongodb_arr":meta.get("mongodb_arr"),
                "sqlitedb_df":meta.get("sqlitedb_df"),
                "st_sqlitedb_arr":meta.get("st_sqlitedb_arr"),
                "st_mongodb_arr":meta.get("st_mongodb_arr"),
                "st_sqlitedb_df": meta.get("st_sqlitedb_df"),
                "generated_chart": meta.get("generated_chart"),
                "timestamp": meta.get("timestamp", 0)
            })

    return sorted(conversations, key=lambda x: x["timestamp"])

def print_conversations():
    conversations = load_conversations()
    for conversation in conversations:
        timestamp = conversation.get("timestamp")  
        if timestamp:
            readable_time = time.ctime(timestamp)  
        else:
            readable_time = "No timestamp available"
        print(f"User: {conversation['user']}, Timestamp: {readable_time}")
        print("-" * 50)  


def delete_conversations():
    try:
        document_ids = chroma_db._collection.get()["ids"]        
        if not document_ids:
            logging.warning("No conversations to delete.")
            return
        
        chroma_db.delete(ids=document_ids)  
        logging.info("All conversations deleted successfully.")
    except Exception as e:
        logging.error(f"Error deleting conversations: {e}")

if __name__ == "__main__":
    df = pd.DataFrame({
    "Region": ["North", "South", "East", "West"],
    "Sales": [10000, 15000, 13000, 9000]
    })
    fig, ax = plt.subplots()
    df.plot(kind="bar", x="Region", y="Sales", ax=ax, color="skyblue")
    
    ax.set_title("Sales Revenue by Region") 
    logging.info("adding question1.")
    add_conversation("question 1", "answer 1",df,fig)
    # logging.info("adding question2.")
    # add_conversation("question 2", "answer 2")
    # logging.info("loading conversations")
    # load_conversations()
    # logging.info("adding question3.")
    # add_conversation("question 3", "answer 3")
    # logging.info("loading conversations.")
    # load_conversations()
    # logging.info("printing conversations.")
    # print_conversations()   
    # logging.info("deleting conversations.")
    # delete_conversations()
    # logging.info("printing conversations.")
    print_conversations()
    # delete_conversations()