import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import json
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ast
from pymongo import MongoClient
import logging
from db_decision_prompt import db_decision_prompt
from sql_query_prompt import sql_query_generating_prompt
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
from mongo_query_prompt import mongodb_quering_prompt
from conversation import add_conversation, delete_conversations, load_conversations
import torch
import io
import base64
from io import StringIO
from bson import ObjectId, Decimal128
import numpy as np
from datetime import datetime  
import time


torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()  
        if isinstance(obj, (ObjectId, Decimal128, pd.Timestamp, np.number, dict, set, bytes)):
            return str(obj)  
        return super().default(obj)  

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

load_dotenv()
api_key = os.getenv("API_KEY")
database_url = os.getenv("DATABASE_FOLDER")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/vector_db_files/chroma_db", embedding_function=embeddings)
model = genai.GenerativeModel('gemini-2.0-flash')
mongo_url = os.getenv("MONGO_URL")
genai.configure(api_key=api_key)  

# =====================================================================================================

def get_llm_response(prompt,question):

    response = model.generate_content([prompt, question])
    response  = response.text.strip()
    return response

def app_start():
    st.markdown("<h1 style='text-align: center;'>CaptenAI Query Genius</h1>", unsafe_allow_html=True)
    st.write("---")

    st.sidebar.image("/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/images/Capten_logo_full.png", width=200)
    st.sidebar.markdown("---")
    st.sidebar.title("Available Databases:")

    with st.sidebar.expander("🏆 **Sports Databases**", expanded=False):
        st.write("""
        - 🏅 **Players** → Stats (goals, assists, matches)
        - 🏆 **Teams** → Rankings & match results
        - 🏟️ **Stadiums** → Locations & capacity
        - 💰 **Financials** → Player salaries & revenue
        """)

    with st.sidebar.expander("🏠 **Real Estate Databases**", expanded=False):
        st.write("""
        - 🏡 **Homes** → Property details & valuation
        - 📍 **Land** → Zoning & land usage
        - ⚡ **Utilities** → Electricity & gas providers
        - 💦 **Water** → Bills & usage
        """)

    with st.sidebar.expander("👨‍💼 **General Databases**", expanded=False):
        st.write("""
        - 🎓 **Students** → Academic performance
        - 💼 **Employees** → Salaries & experience
        - 📈 **Sales** → Revenue statistics
        """)

    with st.sidebar.expander("🗂 **MongoDB Collections**", expanded=False):
        st.write("""
        - 🏡 **Airbnb** → Property reviews & pricing
        - 💳 **Transactions** → Customer analytics
        - 🎬 **Movies** → Cast, reviews & IMDB
        - 🍽️ **Restaurants** → Cuisine & ratings
        """)

    with st.sidebar.expander("🤖 **Vector Database (AI Search)**", expanded=False):
        st.write("""
        - 🏢 **Appstek** → IT & Cloud Analytics
        - 🚀 **Capten AI** → AI Automation & Predictions
        - 🖼️ **ImageVision AI** → Computer Vision & Image Processing
        """)

    st.sidebar.markdown("---")
    st.sidebar.markdown(" **Use this guide to structure your queries efficiently!**")
    st.sidebar.markdown("### Actions:")
    if st.sidebar.button("Delete All Conversations", key="delete_convo"):
        delete_conversations()
        st.session_state.conversation = []  
        st.sidebar.success("All conversations deleted successfully.")
    
    st.sidebar.write("### Current Session Data:")
    st.sidebar.write(st.session_state)


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

# =====================================================================================================

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

# =====================================================================================================
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

# =====================================================================================================

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
        ax.set_xticklabels(df[x_col], rotation=90, ha="right", fontsize=10)  # Rotate for readability
        
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


# =====================================================================================================

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

# =====================================================================================================

def initialize_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = load_conversations()

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

# =====================================================================================================


def main():
    
    initialize_session_state()
    logging.info('*****Started******')
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
                
                
                add_conversation(user_input,summarised_answer,final_response,sqlitedb_arr_json,mongodb_arr_json,sqlitedb_df_json)

# =====================================================================================================

if __name__ == '__main__':
    
    main()