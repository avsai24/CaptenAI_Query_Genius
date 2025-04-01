# CaptenAI Query Genius

![CaptenAI Logo](https://github.com/avsai24/CaptenAI_Query_Genius/blob/main/images/Capten_logo_full.png)  

## 🚀 Overview

**CaptenAI Query Genius** is a powerful Streamlit-based multi-database query retrieval and analysis system. It allows users to retrieve, analyze, and visualize data from **SQLite, MongoDB, and ChromaDB** using **Google Gemini AI** for intelligent query classification and summarization.

### **✨ Features**
- 🔍 **Multi-Database Querying:** Supports **SQLite**, **MongoDB**, and **ChromaDB (VectorDB)**.
- 🤖 **AI-Powered Query Classification:** Uses **Google Gemini AI** to determine the best database for a query.
- 📊 **Automatic Data Summarization & Visualization:** Generates summarized insights and visualizations.
- 🛠 **Session-Based Conversation Tracking:** Stores and retrieves previous queries and responses.
- 🔑 **Secure Authentication:** Uses **bcrypt-encrypted passwords** and **EncryptedCookieManager** for authentication.
- ⚡ **Efficient Query Execution:** Executes SQL and MongoDB queries dynamically based on intent.
- 📁 **User-Specific Chat Histories:** Stores user interactions in **ChromaDB** for personalized insights.
- 🛢 **Account & Chat Management:** Allows users to **delete chats** and **accounts** securely.

---

## 🏰 **Project Structure**
```
📂 captenai-query-genius/
<<<<<<< HEAD
│-- 📝 main.py                     # Main entry point for the Streamlit application
=======
│-- 📝 frontend.py                     # Main entry point for the Streamlit application
>>>>>>> f91976b9a09fe3ccf1740f96d4e380009910e97c
│-- 📝 backend.py                   # Backend functions for query processing, database handling, and AI integration
│-- 📝 db_decision_prompt.py        # Prompt template for AI-based database selection
│-- 📝 sql_query_prompt.py          # Prompt template for generating SQL queries
│-- 📝 mongo_query_prompt.py        # Prompt template for generating MongoDB queries
<<<<<<< HEAD
│-- 📂 database/                    # SQLite database storage
│-- 📂 logs/                         # Application logs
=======
│-- 📂 sqlite_databases/                    # SQLite database storage
│-- 📝 poc.log                       # Application logs
>>>>>>> f91976b9a09fe3ccf1740f96d4e380009910e97c
│-- 📂 vector_db/                    # ChromaDB storage for vector search
│-- 📝 .env                          # Environment variables (API keys, DB paths)
│-- 📝 requirements.txt              # Python dependencies
│-- 📝 README.md                     # Documentation file
```

---

## ⚙️ **Installation & Setup**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/avsai24/CaptenAI_Query_Genius.git
cd captenai-query-genius
```

### **2️⃣ Set Up Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4️⃣ Configure Environment Variables**
Create a `.env` file and add:
```ini
API_KEY="your-gemini-api-key"
MONGO_URL="your-mongodb-connection-string"
DATABASE_FOLDER="path-to-sqlite-database"
CHROMA_DB_PATH="path-to-chromadb"
COOKIE_PREFIX="your-cookie-prefix"
COOKIE_PASSWORD="your-secure-cookie-password"
LOGO_URL="your-logo-url"
```

### **5️⃣ Run the Application**
```sh
streamlit run main.py
```

---

## 🔑 **Authentication**
1. **User Login:** Users must log in using their credentials stored in **SQLite**.
2. **Secure Storage:** Passwords are hashed using **bcrypt**, and session management is handled via **cookies**.
3. **Account Management:** Users can **delete accounts**, **clear chat history**, and **log out** securely.

---

## 🎯 **Usage Guide**
### **1️⃣ Login & Authentication**
- Users must log in or create an account.
- Once authenticated, they can access the main interface.

### **2️⃣ Query Execution**
- Users enter queries in the **chatbox**.
- The system **classifies the query** and selects the appropriate database.
- Results are fetched, **summarized using AI**, and displayed.

### **3️⃣ Data Visualization**
- If the retrieved data supports visualization, a **chart is generated** dynamically.
- **Supported charts:** Bar, Line, Scatter, Histogram, Pie.

### **4️⃣ Chat Management**
- **New Chat:** Start a fresh conversation.
- **Chat History:** Retrieve past queries.
- **Delete Chats:** Remove old conversations.

### **5️⃣ Logout & Account Deletion**
- **Logout:** Ends the session.
- **Delete Account:** Erases all user data from **SQLite and ChromaDB**.

---

## ⚖️ **Security Measures**
- **🔒 Encrypted Passwords:** All user credentials are stored **securely** with **bcrypt hashing**.
- **🔐 Cookie Encryption:** User sessions are protected with **AES encryption**.
- **🛠️ Secure Query Execution:** Prevents SQL injection and ensures safe MongoDB operations.

---

## 🛠 **Tech Stack**
- **Frontend:** Streamlit
- **Databases:** SQLite, MongoDB, ChromaDB
- **AI & Embeddings:** Google Gemini AI, HuggingFace Sentence-Transformers
- **Security:** bcrypt, EncryptedCookieManager
- **Data Visualization:** Matplotlib
- **Backend:** Python, LangChain, Pymongo

---

## 👉 **License**
This project is licensed under the [MIT License](LICENSE).

---

## 👥 **Contributing**
We welcome contributions! Feel free to fork the repo, make changes, and submit a PR.

### **Steps to Contribute**
1. **Fork the Repository**
2. **Create a New Branch**
   ```sh
   git checkout -b feature-branch
   ```
3. **Make Your Changes**
4. **Commit & Push**
   ```sh
   git commit -m "Your changes"
   git push origin feature-branch
   ```
5. **Open a Pull Request**

---

## 💌 **Contact & Support**
If you have any questions, feel free to reach out:

📧 Email: venkatasaiancha24@gmail.com  
💻 LinkedIn: https://www.linkedin.com/in/venkatasaiancha/  
📂 GitHub: https://github.com/avsai24  

---

🚀 **CaptenAI Query Genius - Your AI-Powered Multi-Database Assistant!** 🌟

