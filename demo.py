import streamlit as st
import sqlite3
import bcrypt
import time
import re
from streamlit_cookies_manager import EncryptedCookieManager
from main import main
from conversation import delete_collection

LOGO_URL = "/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/images/Capten_logo_full.png"
cookies = EncryptedCookieManager(prefix="query_genius", password="appstek")

if not cookies.ready():
    st.stop()

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
    cookies.save()

    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    if "conversation" in st.session_state:
        del st.session_state["conversation"]
    st.rerun()

def check_session():
    expires = cookies.get("expires")
    if expires and float(expires) > time.time():
        st.session_state["authenticated"] = True
        st.session_state["username"] = cookies.get("username")
    else:
        st.session_state["authenticated"] = False

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
    status_placeholder.success(" User removed from SQLite.")

    time.sleep(1)
    delete_collection()
    progress_bar.progress(60)
    status_placeholder.success("Data deleted from VectorDB.")

    time.sleep(1)
    progress_bar.progress(90)
    status_placeholder.warning("Finalizing account deletion...")

    time.sleep(1)
    progress_bar.progress(100)
    status_placeholder.success(f" Account '{username}' deleted successfully!")

    time.sleep(2) 
    status_placeholder.info("Redirecting to main page...")
    
    time.sleep(1)  
    logout() 

def main_func():
    # st.write(st.session_state)
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

        with st.container():
            st.image(LOGO_URL)

            if "show_signup" not in st.session_state:
                st.session_state["show_signup"] = False

            if not st.session_state["show_signup"]:
                st.subheader("Login to Your Account")
                with st.form("login_form"):
                    username = st.text_input("Username", key="login_username", placeholder="Enter your username")
                    password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
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
                    new_username = st.text_input("Choose a Username", key="signup_username", placeholder="Your unique username")
                    new_password = st.text_input("Choose a Password", type="password", key="signup_password", placeholder="At least 8 characters")
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
        
        main()
        
        if st.sidebar.button("Delete Account"):
            delete_account()
        
        if st.sidebar.button("Logout"):
            logout()






if __name__ == '__main__':
   main_func()