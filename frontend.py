import os
from backend import *
from dotenv import load_dotenv
from streamlit_cookies_manager import EncryptedCookieManager

load_dotenv()
prefix = os.getenv("COOKIE_PREFIX")
password = os.getenv("COOKIE_PASSWORD")
LOGO_URL = os.getenv("LOGO_URL")

setup_logging()
cookies = EncryptedCookieManager(prefix=prefix, password=password)
create_users_db()

logging.info("App Starts")
if not cookies.ready():
    st.stop()

check_session(cookies)
markdown_for_buttons()

if not st.session_state.get("authenticated"):
    login_form(LOGO_URL, cookies)
else:
    logging.info("user authenticated")
    initialize_session_state(cookies)
    app_start(LOGO_URL,cookies)
    display_conversation_history() 
    backend_func()

logging.info("App ends")
