import streamlit as st
import bcrypt
import re
import sqlite3
import pandas as pd
import json
import logging
import warnings
import io
import base64
from io import StringIO
import time
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ast
from pymongo import MongoClient
import logging
from db_decision_prompt import db_decision_prompt

from sql_query_prompt import sql_query_generating_prompt
from mongo_query_prompt import mongodb_quering_prompt
import torch
from bson import ObjectId, Decimal128
import numpy as np
from datetime import datetime  
import pprint 
from langchain.schema import Document
import shutil
from chromadb.config import Settings
warnings.filterwarnings("ignore", category=UserWarning) 
load_dotenv()


torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

api_key = os.getenv("API_KEY")
database_url = os.getenv("DATABASE_FOLDER")
mongo_url = os.getenv("MONGO_URL")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
vectordb_path = os.getenv("VECTORDB_PATH")
genai.configure(api_key=api_key)  
chroma_settings = Settings(anonymized_telemetry=False)

@st.cache_resource
def initialise_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def initialise_model():
    return genai.GenerativeModel('gemini-2.0-flash')

@st.cache_resource
def initialise_vectordb():
    embeddings = initialise_embeddings()
    return Chroma(persist_directory=vectordb_path, embedding_function=embeddings,client_settings=chroma_settings)

vectorstore =initialise_vectordb()
model = initialise_model()




# =====================================================================================================
# vectordb retrieving functions

def delete_folders_except_sqlite(parent_directory):
    
    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)

        if os.path.isdir(item_path):  
            shutil.rmtree(item_path)

def get_user_collection():
    
    if "username" not in st.session_state:
        logging.error("User is not logged in. Cannot create ChromaDB collection.")
        return None
    embeddings = initialise_embeddings()
    username = st.session_state["username"]
    collection_name = f"user_{username}"  
    logging.info(f"Using ChromaDB collection: {collection_name}")

    return Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=CHROMA_DB_PATH,client_settings=chroma_settings)

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


# =====================================================================================================
# frontend

def setup_logging():
    
    if logging.getLogger().hasHandlers():
        return

    file_handler = logging.FileHandler("poc.log", mode="a")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

@st.cache_resource
def create_users_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

create_users_db()

def is_strong_password(password):
    
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must contain at least one number."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return "Password must contain at least one special character (!@#$%^&* etc.)."
    
    return None 

def add_user(username, password):

    error_msg = is_strong_password(password)
    if error_msg:
        st.error(f"{error_msg}")
        

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        st.success("Account created successfully!.")
        st.session_state["show_signup"] = False 
        login_user(username)

    except sqlite3.IntegrityError:
        st.error("Username already exists. Try another.")

    conn.close()

def verify_login(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username=?", (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result and bcrypt.checkpw(password.encode(), result[0].encode()):
        return True
    return False

def delete_user(username):
    conn = sqlite3.connect("users.db")  
    cursor = conn.cursor()

    cursor.execute("DELETE FROM users WHERE username = ?", (username,))

    conn.commit()
    conn.close()

    print(f"User '{username}' deleted successfully.")

def delete_account():
    username = st.session_state["username"]
    
    status_placeholder = st.empty()  
    progress_bar = st.progress(0)  

    status_placeholder.info("Deleting your account... Please wait.")

    time.sleep(1)
    delete_user(username)
    progress_bar.progress(30)
    status_placeholder.success("User removed from SQLite.")

    time.sleep(1)
    delete_collection()
    progress_bar.progress(60)
    status_placeholder.success("Data deleted from VectorDB.")
    
    time.sleep(1)
    progress_bar.progress(90)
    status_placeholder.warning("Finalizing account deletion...")

    time.sleep(1)
    progress_bar.progress(100)
    status_placeholder.success(f"Account '{username}' deleted successfully!")

    time.sleep(2) 
    status_placeholder.info("Redirecting to main page...")

    time.sleep(1)
    
    logout()

def login_user(username,cookies):
    cookies["username"] = username
    cookies["logged_in"] = "true"
    cookies["expires"] = str(time.time() + 3600)  
    cookies.save()

    st.session_state["authenticated"] = True
    st.session_state["username"] = username

    time.sleep(0.5)
    st.rerun()

def logout(cookies):
    cookies["username"] = ""
    cookies["logged_in"] = "false"
    cookies["expires"] = str(time.time()) 
    cookies["chat_id"] = ""
    print(f'while logout: {cookies["chat_id"]}')
    cookies.save()

    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    if "chat_id" in st.session_state:
        del st.session_state["chat_id"]
    if "conversation" in st.session_state:
        del st.session_state["conversation"]
    st.rerun()

def check_session(cookies):
    logging.info("entered check session")
    expires = cookies.get("expires")
    if expires and float(expires) > time.time():
        st.session_state["authenticated"] = True
        st.session_state["username"] = cookies.get("username")
    else:
        st.session_state["authenticated"] = False
    chat_id = cookies.get("chat_id")
    
    if chat_id and str(chat_id).strip().isdigit():
        chat_id = int(chat_id)
    else:
        chat_id = None  

    st.session_state["chat_id"] = chat_id
    logging.info(f"check_session(): Retrieved chat_id from cookies -> {chat_id}") 

def app_start(LOGO_URL,cookies):
    logging.info("entered app start")
    
    st.sidebar.image(LOGO_URL, width=200)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
    f"""
    <div style="
        padding: 10px; 
        border-radius: 8px; 
        background-color: #f0f2f6;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        color: #333;">
        👤 <b>Current User:</b> {st.session_state.get('username', 'Guest')}
    </div>
    """,
    unsafe_allow_html=True
    )
    st.sidebar.write("---")
    if st.sidebar.button("New Chat"):
        chat_id = get_new_chat_id()
        st.session_state.chat_id = chat_id
        st.session_state["conversation"] = get_chats_by_id(chat_id)
        cookies["chat_id"] = str(chat_id) 
        cookies.save()
        st.rerun()

    st.sidebar.markdown("##  Chat History")
    chat_ids = get_chat_ids()
    
    if chat_ids:
        with st.sidebar.expander("View Chats", expanded=True):
            for chat_id in chat_ids:
                if st.button(f" Chat {chat_id}"):
                    st.session_state["chat_id"] = chat_id
                    st.session_state["conversation"] = get_chats_by_id(chat_id)
                    cookies["chat_id"] = str(chat_id) 
                    cookies.save()

                    logging.info(f"Selected chat_id: {chat_id}") 

                    st.rerun()

    else:
        st.sidebar.info("No chat history available.")

    st.sidebar.markdown("## Actions")
    
    if st.sidebar.button("Delete Current chat"):
        chat_id = st.session_state.get("chat_id")
        if st.session_state["conversation"]:
            delete_chat_by_id(chat_id)
            new_chat_id = get_new_chat_id()
            st.session_state["chat_id"] = new_chat_id
            st.session_state["conversation"] = get_chats_by_id(new_chat_id)
            cookies["chat_id"] = str(new_chat_id) 
            cookies.save()

            st.sidebar.success("Current chat is successfully deleted!!")
            time.sleep(2)
            st.rerun()

        else:
            st.sidebar.error("No conversation to delete!!")
        

    if st.sidebar.button("Logout"):
            logout(cookies)
    
    if "confirm_delete" not in st.session_state:
        st.session_state["confirm_delete"] = False

    if st.sidebar.button("Delete Account"):
        st.session_state["confirm_delete"] = True 

    if st.session_state["confirm_delete"]:
        st.sidebar.warning("Are you sure you want to delete your account? This action cannot be undone.")

        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("Cancel"):
                st.session_state["confirm_delete"] = False  
                st.rerun()  

        with col2:
            if st.button("Confirm Deletion", type="primary"):
                delete_account()
    
    # st.sidebar.write("### Current Session Data:")
    # st.sidebar.write(st.session_state)

def initialize_session_state(cookies):
    logging.info("entered initialise session state")
    if "chat_id" not in st.session_state or not st.session_state["chat_id"]:
        new_chat_id = get_new_chat_id()  
        st.session_state["chat_id"] = new_chat_id
        cookies["chat_id"] = str(new_chat_id)  
        cookies.save()

    chat_id = st.session_state.chat_id
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_chats_by_id(chat_id)

def display_conversation_history():
    logging.info("entered display_conversation history")
    
    for entry in st.session_state.conversation:

        with st.chat_message("user"):
            st.markdown(f"{entry['user_input']}") 
    
        with st.chat_message("assistant"):
            st.markdown(f"{entry['model_response']}")

            if not entry['final_response']:
                if entry['generated_chart'] and not entry['st_sqlitedb_df'].empty:
                    st.table(entry['st_sqlitedb_df'])
                    decoded_bytes = base64.b64decode(entry['generated_chart'])
                    st.image(io.BytesIO(decoded_bytes), caption="generated chart Chart")
                else:
                    sqlitedb_arr = entry['st_sqlitedb_arr']
                    mongodb_arr = entry['st_mongodb_arr']
                    if len(mongodb_arr)!=0:
                        for df in mongodb_arr:
                            st.write(df)
                    if len(sqlitedb_arr)!=0:
                        for df in sqlitedb_arr:
                            st.write(df)
            else:
                
                hist_final_response_dict = {}
                if entry.get("final_response"):
                    hist_final_response_dict = string_response_to_json(entry["final_response"])

                hist_sqlitedb_df = None
                if entry.get("sqlitedb_df") and entry["sqlitedb_df"].strip():
                    try:
                        hist_sqlitedb_df = pd.read_json(StringIO(entry["sqlitedb_df"]))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding sqlitedb_df JSON: {e}")
                    except Exception as e:
                        print(f"Unexpected error parsing sqlitedb_df: {e}")

                if entry["sqlitedb_arr"] and entry["sqlitedb_arr"] != "[]":
                    sqlitedb_arr_python_list = json.loads(entry["sqlitedb_arr"])

                    hist_sqlitedb_arr = [pd.DataFrame(json.loads(json_str)) for json_str in sqlitedb_arr_python_list]
                else:
                    hist_sqlitedb_arr = []  
                
                hist_mongodb_arr = convert_json_list_to_dataframe(entry["mongodb_arr"]) 

                if hist_final_response_dict.get("visualization_required") == "Yes" and hist_sqlitedb_df is not None:
                    chart_type = hist_final_response_dict.get("chart_type")
                    columns = hist_final_response_dict.get("columns_for_visualization")
                    chart_type = chart_type_determine(chart_type)
                    
                    if hist_sqlitedb_df is not None:
                        st.table(hist_sqlitedb_df)
                    else:
                        if len(hist_sqlitedb_arr)!=0:
                            for df in hist_sqlitedb_arr:
                                st.write(df)
                        if len(hist_mongodb_arr)!=0:
                            for df in hist_mongodb_arr:
                                st.write(df)
                        
                        
                    if chart_type and columns:
                        gen_chart = generate_chart(hist_sqlitedb_df, chart_type, columns)
                        if gen_chart:    
                            st.pyplot(gen_chart)
                else:
                    answer = entry['model_response']
                    if "Insufficient data to answer" not in answer:
                        
                        if len(hist_mongodb_arr)!=0:
                            for df in hist_mongodb_arr:
                                st.write(df)
                        if len(hist_sqlitedb_arr)!=0:
                            for df in hist_sqlitedb_arr:
                                st.write(df)

def markdown_for_buttons():
    st.markdown(
    """
    <style>
    .stTextInput>div>div>input {
        padding: 10px;
        font-size: 18px;
    }
    .stButton>button {
        padding: 10px;
        font-size: 18px;
        width: 100%;
    }
    .container {
        width: 50%;
        margin: auto;
        padding: 2rem;
        border-radius: 10px;
        background-color: #f9f9f9;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #4a4a4a;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

def login_form(LOGO_URL,cookies):
    logging.info("user has not authenticated")
    with st.container():
        st.image(LOGO_URL)

        if "show_signup" not in st.session_state:
            st.session_state["show_signup"] = False

        if not st.session_state["show_signup"]:
            st.subheader("Login to Your Account")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                login_button = st.form_submit_button("Login ")

                if login_button:
                    if verify_login(username, password):
                        login_user(username,cookies)
                        st.rerun()
                    else:
                        st.error(" Invalid username or password!")

            st.markdown("---")
            if st.button("Create an Account", help="Click to sign up"):
                st.session_state["show_signup"] = True
                st.rerun()

        else:
            st.subheader("Create an Account")
            with st.form("signup_form"):
                new_username = st.text_input("Choose a Username", placeholder="Your unique username")
                new_password = st.text_input("Choose a Password", type="password", placeholder="At least 8 characters")
                create_button = st.form_submit_button("Sign Up")

                if create_button:
                    if new_username and new_password:
                        add_user(new_username, new_password)
                    else:
                        st.error(" Username and password cannot be empty.")

            st.markdown("---")
            if st.button("Back to Login", help="Go back to the login screen"):
                st.session_state["show_signup"] = False
                st.rerun()

def backend_func():

    if user_input := st.chat_input("Type your message..."):
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Model fetching Data from Databases. Please Wait..."):
            logging.info("Retriving docs from vector store")
            vectordb_arr = retrieve_docs(user_input)
            logging.info("Done retriving relavent documents from vector store.")
            logging.info('LLm deciding databases to search.') 
            db_decision = get_llm_response(db_decision_prompt,user_input)
            
            logging.info("Completed deciding database.")
            
            
            initial_db_decision = string_response_to_json(db_decision)
            
            if isinstance(initial_db_decision, dict):
                sqlitedb_arr = []
                mongodb_arr = []
                
                if initial_db_decision.get("queries"):
                    for result in initial_db_decision["queries"]:
                        initial_database = result["database"]
                        initial_question = result["question"]
                        if initial_database == "sqliteDB":
                            logging.info("Querying Sqlite database.")
                            sqliteDb_response = get_sqliteDB_response(initial_question)

                            if sqliteDb_response is not None and not sqliteDb_response.empty:
                                sqlitedb_arr.append(sqliteDb_response)
                        
                        if initial_database == "mongodb":
                            logging.info("Querying mongodb database.")
                            mongoDb_response = get_mongoDB_response(initial_question)
                            if mongoDb_response is not None and not mongoDb_response.empty:
                                mongodb_arr.append(mongoDb_response)

                sqlitedb_df = None
                if sqlitedb_arr:  
                    sqlitedb_df = merge_arr(sqlitedb_arr)

                
                logging.info("getting final summarized response from llm")
                final_response = final_response_from_llm(
                sqlitedb_df ,
                sqlitedb_arr, 
                vectordb_arr,
                mongodb_arr,
                user_input
                )
                
                
                final_response_dict = string_response_to_json(final_response)

                chart_base64 = None 

                if final_response_dict.get("visualization_required") == "Yes" and sqlitedb_df is not None:
                    summarised_answer = final_response_dict.get("summarized_answer")
                    
                    with st.chat_message("assistant"):
                        st.write_stream(generate_response(summarised_answer))
                        chart_type = final_response_dict.get("chart_type")
                        columns = final_response_dict.get("columns_for_visualization")
                        chart_type = chart_type_determine(chart_type)
                        if sqlitedb_arr is not None:
                            st.table(sqlitedb_df)
                        else:
                            if len(sqlitedb_arr)!=0:
                                for df in sqlitedb_arr:
                                    st.write(df)
                            if len(mongodb_arr)!=0:
                                for df in mongodb_arr:
                                    st.write(df)
                            
                            
                        if chart_type and columns:
                            
                            gen_chart = generate_chart(sqlitedb_df, chart_type, columns)
                            
                            if gen_chart:
                                st.pyplot(gen_chart)
                        
                                img_buffer = io.BytesIO()
                                gen_chart.savefig(img_buffer, format="png")
                                img_buffer.seek(0)
                                chart_bytes = img_buffer.getvalue()
                                chart_base64 = base64.b64encode(chart_bytes).decode("utf-8")

                        st.session_state.conversation.append({"user_input": user_input,
                                                            "model_response": summarised_answer,
                                                            "final_response": None,
                                                            "sqlitedb_arr": None,
                                                            "mongodb_arr": None,
                                                            "sqlitedb_df":None,
                                                            "st_sqlitedb_arr": sqlitedb_arr,
                                                            "st_mongodb_arr":mongodb_arr,
                                                            "st_sqlitedb_df":sqlitedb_df,
                                                            "generated_chart":chart_base64})
                        logging.info('*****ended******')
                        
                elif final_response_dict.get("error"):
                    with st.chat_message("assistant"):
                            summarised_answer = final_response_dict.get("error")
                            st.write(summarised_answer)
                            st.session_state.conversation.append({"user_input": user_input,
                                                                "model_response": summarised_answer,
                                                                "final_response": None,
                                                                "sqlitedb_arr": None,
                                                                "mongodb_arr": None,
                                                                "sqlitedb_df":None,
                                                                "st_sqlitedb_arr": sqlitedb_arr,
                                                                "st_mongodb_arr":mongodb_arr,
                                                                "st_sqlitedb_df":None,
                                                                "generated_chart":None})
                else:
                    
                    summarised_answer = final_response_dict.get("summarized_answer")
                    
                    if "Insufficient data to answer" not in summarised_answer:
                        with st.chat_message("assistant"):
                            st.write_stream(generate_response(summarised_answer))
                            
                            if len(mongodb_arr)!=0:
                                for df in mongodb_arr:
                                    st.write(df)
                            if len(sqlitedb_arr)!=0:
                                for df in sqlitedb_arr:
                                    st.write(df)

                        st.session_state.conversation.append({"user_input": user_input,
                                                                "model_response": summarised_answer,
                                                                "final_response": None,
                                                                "sqlitedb_arr": None,
                                                                "mongodb_arr": None,
                                                                "sqlitedb_df":None,
                                                                "st_sqlitedb_arr": sqlitedb_arr,
                                                                "st_mongodb_arr":mongodb_arr,
                                                                "st_sqlitedb_df":None,
                                                                "generated_chart":None})
                        logging.info('*****ended******')
                    else:
                        with st.chat_message("assistant"):
                            st.write(summarised_answer)
                
                if sqlitedb_df is not None and not sqlitedb_df.empty:
                    sqlitedb_df_json = sqlitedb_df.to_json(orient="records")
                else:
                    sqlitedb_df_json = "None" 

                sqlitedb_arr_json_list = [
                    df.to_json(orient="records") for df in sqlitedb_arr if isinstance(df, pd.DataFrame) and not df.empty
                ]
                sqlitedb_arr_json = json.dumps(sqlitedb_arr_json_list) if sqlitedb_arr_json_list else "[]"

                
                mongodb_arr_json = convert_dataframe_to_json_list(mongodb_arr)
                
                
                add_conversation_to_existing_chat(user_input,summarised_answer,final_response,sqlitedb_arr_json,mongodb_arr_json,sqlitedb_df_json)

# =====================================================================================================

# LLM functions

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()  
        if isinstance(obj, (ObjectId, Decimal128, pd.Timestamp, np.number, dict, set, bytes)):
            return str(obj)  
        return super().default(obj) 

def get_llm_response(prompt,question):

    response = model.generate_content([prompt, question])
    response  = response.text.strip()
    return response

def retrieve_docs(user_input):
    docs = vectorstore.similarity_search_with_relevance_scores(user_input, k=10)
    THRESHOLD = 0.25
    useful_docs = [(doc, score) for doc, score in docs if score >= THRESHOLD]
    return useful_docs

def string_response_to_json(json_string):
    if not json_string or json_string.strip() == "":
        st.error("Error: The string is empty or None.")
        return {"error": "Empty response received."}
    
    if json_string.startswith("```json") and json_string.endswith("```"):
        json_string = json_string[7:-3].strip()

    try:
        return json.loads(json_string) 
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return {"error": "Invalid JSON format."}

#-------------------------

def dynamic_query_executor(sql_query, primary_db):
    db_path = os.path.join(database_url, primary_db)

    if not os.path.exists(db_path):
        st.error(f"Error: Database '{primary_db}' not found in '{database_url}'.")
        return None
    try:
        conn = sqlite3.connect(db_path)
        
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        if df.empty:    
            return None
        return df

    except sqlite3.Error as e:
        return None
    except Exception as e:
        return None

def merge_arr(arr):
    if len(arr) == 1:
        return arr[0]  
    
    merged_df = arr[0] 

    for df in arr[1:]:
        common_cols = list(set(merged_df.columns) & set(df.columns)) 

        if common_cols:
            merged_df = pd.merge(merged_df, df, on=common_cols[0], how='inner')  
        else:
            return None
    return merged_df

def get_sqliteDB_response(question):
    arr=[]
    
    response = get_llm_response(sql_query_generating_prompt,question)
    if "queries" not in response:
        df = pd.DataFrame([])
        return df
    else: 
        response_dict = string_response_to_json(response)
        for result in response_dict["queries"]:
            database = result["database"]
            sql_query = result["sql"]
            df = dynamic_query_executor(sql_query, database)
            if df is not None:
                arr.append(df)
        if len(arr)!=0:
            final_df= merge_arr(arr)
            return final_df

#-------------------------

def connect_to_mongodb():
    try:
        client = MongoClient(mongo_url)
        logging.info("Connected to mongodb")
        return client
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return None

def parse_response_dict(response_dict):
    
    if isinstance(response_dict, dict):
        database = response_dict.get("database")
        collection = response_dict.get("collection")
        if response_dict.get("query_filter"):
            query_filter = response_dict.get("query_filter")
        else:
            query_filter = None
        if response_dict.get("projection"):
            projection = response_dict.get("projection")
        else:
            projection = None
        if response_dict.get("sort"):
            sort = response_dict.get("sort")
        else:
            sort = None
        if response_dict.get("limit"):
            limit = response_dict.get("limit")
        else:
            limit = 5

        return database, collection, query_filter, projection, sort, limit
    else:
        database= None
        collection=None
        query_filter=None
        projection=None
        sort= None
        limit=None
        return database, collection, query_filter, projection, sort, limit

def execute_mongodb_query(client,database, collection, query_filter, projection, sort, limit):

    try:
        db = client[database]
        col = db[collection]

        if query_filter and isinstance(query_filter, str):
            query_filter = ast.literal_eval(query_filter) 
        if projection and isinstance(projection, str):
            projection = ast.literal_eval(projection) 
        if sort and isinstance(sort, str):
            sort = ast.literal_eval(sort) 

        cursor = col.find(query_filter, projection)

        if sort:
            cursor = cursor.sort(sort)

        if limit:
            cursor = cursor.limit(limit)

        result = list(cursor)
        df = pd.DataFrame(result)
        
        return df
    
    except Exception as e:
        print(f"Error executing MongoDB query: {e}")
        return None

def get_mongoDB_response(question):
    client = connect_to_mongodb()
    logging.info("getting response from llm")
    
    response = get_llm_response(mongodb_quering_prompt, question)
    
    response_dict = string_response_to_json(response)
    
    if response_dict.get("error"):
        df = pd.DataFrame([])
        logging.info("mongodb returning empty df")
        return df
    else:
        database, collection, query_filter, projection, sort, limit = parse_response_dict(response_dict)

        df = execute_mongodb_query(client,database, collection, query_filter, projection, sort, limit)
        logging.info("mongodb returning answer")
        
        return df

#------------------------------

def final_response_from_llm(sqlitedb_df,sqlitedb_arr,vectordb_arr,mongodb_arr,user_input ):
    prompt = f"""

    You are an intelligent summarization and visualization assistant. Given a user question and retrieved data from multiple sources (SQLite, MongoDB, and VectorDB), your task is to:

    Summarize the answer strictly based on the retrieved data.

    Determine if visualization is meaningful and recommend an appropriate chart type.

    Input Format

    User Question: {user_input}

    Retrieved Data:

    SQLite DataFrame (Merged SQLite Data): {sqlitedb_df}

    All SQLite DataFrames: {sqlitedb_arr}

    MongoDB Data: {mongodb_arr}

    VectorDB Data: {vectordb_arr}

    Strict Summarization Rules

    Use Only Retrieved Data

    Summarize strictly from {sqlitedb_df}, {sqlitedb_arr}, {mongodb_arr}, and {vectordb_arr}.

    Do not hallucinate, infer missing data, or generate information beyond available data.

    If all retrieved data sources are empty, explicitly return:

    {{ "error": "Insufficient data to answer." }}

    Handling Unrelated Multi-Database Queries

    If retrieved data comes from multiple unrelated databases, return:

    {{ "error": "Your question is not related between databases." }}

    Concise Answering

    If the dataset fully answers the question, provide a short, precise summary.

    If multiple relevant answers exist, summarize key insights clearly.

    Visualization Rules

    Check if Visualization is Required

    If only one row of data exists, visualization is not needed:

    {{ "visualization_required": "No", "chart_type": null, "direct_answer": "Not enough data points for visualization." }}

    If aggregate values (e.g., max salary, min salary) exist with a single data point, visualization is not needed.

    If at least two rows exist, determine a valid chart type.

    Check for Numeric Data Before Suggesting a Chart

    If both columns are categorical, return:

    {{ "visualization_required": "No", "chart_type": null }}

    If at least one column is numeric, suggest a valid chart type.

    If no numeric data is available, return:

    {{ "visualization_required": "No", "chart_type": null, "direct_answer": "No numerical data available for visualization." }}

    Chart Selection Rules

    Histograms: Require exactly one numeric column.

    If categorical columns are selected, return Bar Chart instead.

    Bar Charts: Require one categorical (X-axis) and one numeric column (Y-axis).

    Scatter Plots: Require two numeric columns.

    Line Charts: Require one categorical/time-based X-axis and one numeric Y-axis.

    Pie Charts: Require one categorical (X-axis) and one numeric column (Y-axis).

    If there are more than 10 unique categories, return:

    {{ "visualization_required": "No", "direct_answer": "Too many categories for pie chart." }}

    Expected JSON Response Format

    {{
    "summarized_answer": "<Concise answer based on available data>",
    "visualization_required": "<Yes/No>",
    "chart_type": "<Suggested chart type or null>",
    "columns_for_visualization": ["<Column1>", "<Column2>"] or null,
    "direct_answer": "<Optional: Extra message if visualization is not possible>"
    }}

    Example Scenarios & Expected Responses

    Case 1: No Data Available (Error Case)

    User Question: "What is the total revenue generated by sales in the last quarter?"

    Retrieved Data: {{}}

    AI Response:

    {{ "error": "Insufficient data to answer." }}

    Case 2: Direct Answer, No Visualization

    User Question: "What is the average salary of employees?"

    Final DataFrame:

    Employee  Salary
    A        60000
    B        75000
    C        80000

    AI Response:

    {{
        "summarized_answer": "The average salary of employees is $71,667.",
        "visualization_required": "No",
        "chart_type": null,
        "columns_for_visualization": null
    }}

    Case 3: Visualization Recommended

    User Question: "How does salary vary across departments?"

    Final DataFrame:

    Department  Salary
    HR         50000
    IT         90000
    Sales      70000

    AI Response:

    {{
        "summarized_answer": "The dataset contains salary information categorized by department.",
        "visualization_required": "Yes",
        "chart_type": "Bar Chart",
        "columns_for_visualization": ["Department", "Salary"]
    }}

    Final Fixes & Improvements

    ✅ Ensures JSON validity for all cases.✅ Strictly returns an error if no data is retrieved.✅ Prevents incorrect visualizations when only one row exists.✅ Ensures visualization is meaningful by requiring at least two rows.✅ Prevents charts with purely categorical data.✅ Strictly enforces accurate summarization using only retrieved data.✅ Handles unrelated multi-database queries explicitly.✅ Guarantees consistent output for repeated questions.

    This refined prompt ensures structured, accurate, and meaningful responses for summarization and visualization, without hallucination. 🚀


"""
    
    response = model.generate_content(prompt)
    response  = response.text.strip()
    return response

def chart_type_determine(chart_type):
    if "bar" in chart_type.lower():
        return "bar"
    elif "line" in chart_type.lower():
        return "line"
    elif "scatter" in chart_type.lower():
        return "scatter"
    elif "histogram" in chart_type.lower():
        return "histogram"
    elif "pie" in chart_type.lower():
        return "pie"
    else:
        return None

def generate_chart(df, chart_type, columns):
    if not columns or not isinstance(columns, list) or len(columns) == 0:
        st.error("Error: No valid columns provided for visualization!")
        return None

    df.columns = df.columns.str.strip()
    columns = [col.strip() for col in columns]

    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        logging.info(f"Error: The following columns are missing in DataFrame: {missing_cols}")
        return None

    x_col = columns[0]
    y_col = columns[1] if len(columns) > 1 else None  

    if x_col not in df.columns:
        st.error(f"Error: Column '{x_col}' not found in DataFrame.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))  

    
    if chart_type.lower() == "histogram":
        if not pd.api.types.is_numeric_dtype(df[x_col]):
            st.error(f"Error: Column '{x_col}' is not numeric. Cannot plot histogram.")
            return None

        ax.hist(df[x_col], bins=10, color='blue', edgecolor='black')
        ax.set_title(f"Distribution of {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Frequency")

    
    elif chart_type.lower() == "bar":
        if y_col is None:
            st.error("Error: Bar chart requires both X and Y columns.")
            return None

        num_labels = len(df[x_col].unique())

        ax.bar(df.index, df[y_col], color='skyblue')

        ax.set_title(f"{y_col} by {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        ax.set_xticks(df.index)  
        ax.set_xticklabels(df[x_col], rotation=90, ha="right", fontsize=10)  
        
    elif chart_type.lower() == "line":
        if y_col is None:
            st.error("Error: Line chart requires both X and Y columns.")
            return None

        ax.plot(df[x_col], df[y_col], marker='o', linestyle='-', color='red')
        ax.set_title(f"Trend of {y_col} over {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    
    elif chart_type.lower() == "scatter":
        if y_col is None:
            st.error("Error: Scatter plot requires both X and Y columns.")
            return None

        ax.scatter(df[x_col], df[y_col], color='blue')
        ax.set_title(f"{x_col} vs {y_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    elif chart_type.lower() == "pie":
        if y_col is None:
            st.error("Error: Pie chart requires a category and a numeric column.")
            return None

        unique_categories = df[x_col].nunique()
        if unique_categories > 10:
            st.error("Error: Pie chart requires ≤10 unique categories.")
            return None

        df_grouped = df.groupby(x_col)[y_col].sum()
        df_grouped.plot(kind="pie", autopct="%1.1f%%", ax=ax, startangle=90, cmap="Set3")
        ax.set_ylabel("") 
        ax.set_title(f"Distribution of {y_col}")

    else:
        st.error("Error: Unsupported chart type!")
        return None

    return fig  

#------------------------------

def convert_dataframe_to_json_list(dataframes):
    json_list = []

    for df in dataframes:
        if isinstance(df, pd.DataFrame) and not df.empty:
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].apply(lambda x: x.encode("utf-8", "ignore").decode("utf-8") if isinstance(x, str) else x)

            for col in df.columns:
                df[col] = df[col].apply(
                    lambda x: x.isoformat() if isinstance(x, datetime)  
                    else str(x) if isinstance(x, (ObjectId, Decimal128, pd.Timestamp, np.number, dict, set, bytes)) 
                    else x
                )

            for col in df.select_dtypes(include=["object"]).columns:
                if df[col].apply(lambda x: isinstance(x, list)).any():
                    df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [x])

            
            df.fillna("None", inplace=True)

            
            json_string = json.dumps(df.to_dict(orient="records"), ensure_ascii=False, cls=CustomJSONEncoder)
            json_list.append(json_string)

    return json.dumps(json_list, cls=CustomJSONEncoder) if json_list else "[]"

def convert_json_list_to_dataframe(json_string):
    try:
       
        python_list = json.loads(json_string)  
        
        if not python_list:
            return []

        df_list = [pd.DataFrame(json.loads(json_str)) for json_str in python_list]

        for df in df_list:
            for col in df.select_dtypes(include=["object"]).columns:
                if df[col].apply(lambda x: isinstance(x, list)).any(): 
                    df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [x])

        return df_list

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"Error decoding JSON: {e}")
        return []                    

def generate_response(response):
    for line in response.split("\n"):  
        words = line.split()  

        if line.strip().startswith("-") or line.strip().startswith("*") or line.strip().startswith("•"):  
            yield "\n" + line[:2]  
            words = words[1:] 

        for word in words:
            yield word + " "  
            time.sleep(0.05) 

        yield "\n"  







