import streamlit as st
import sqlite3
import bcrypt
import time
import re
from streamlit_cookies_manager import EncryptedCookieManager
from conversation import delete_collection
import streamlit as st
import sqlite3
import pandas as pd
import json
import logging
from db_decision_prompt import db_decision_prompt
import warnings
import io
import base64
from io import StringIO
import time
from main import *
from vector_db_test import *

warnings.filterwarnings("ignore", category=UserWarning) 

load_dotenv()
prefix = os.getenv("COOKIE_PREFIX")
password = os.getenv("COOKIE_PASSWORD")
LOGO_URL = os.getenv("LOGO_URL")
# st.session_state



cookies = EncryptedCookieManager(prefix=prefix, password=password)



for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

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

# =====================================================================================================
if not cookies.ready():
    st.stop()


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

# ----------------------------------------------------------------------------------------------------------------------------
def login_user(username):
    cookies["username"] = username
    cookies["logged_in"] = "true"
    cookies["expires"] = str(time.time() + 3600)  
    cookies.save()

    st.session_state["authenticated"] = True
    st.session_state["username"] = username

    time.sleep(0.5)
    st.rerun()

def logout():
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

def check_session():
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


def app_start():
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
            logout()
    
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

def initialize_session_state():
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




def main_func():
    logging.info("App Starts")
    check_session()
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

    if not st.session_state.get("authenticated"):
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
                            login_user(username)
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
    else:
        logging.info("user authenticated")
        initialize_session_state()
        
        app_start()
        display_conversation_history() 

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
        
    
    logging.info("App ends")






if __name__ == '__main__':
    main_func()
    