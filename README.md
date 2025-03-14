# **CaptenAI Query Genius 🚀**
### **AI-Powered Multi-Database Query Retrieval System**

![CaptenAI Logo](https://github.com/avsai24/CaptenAI_Query_Genius/blob/main/images/Capten_logo_full.png)  

CaptenAI Query Genius is an **AI-driven query system** that intelligently retrieves and processes data from **SQLite, MongoDB, and a vector database**. It leverages **Google Gemini AI** for smart query classification, structured/unstructured data retrieval, and visualization of results.  

---

## **📌 Features**  

✅ **Smart Query Routing** – Automatically selects between **SQLite, MongoDB, or a Vector Database** for query execution.  
✅ **AI-Powered Query Classification** – Uses **Google Gemini AI** to understand and optimize queries.  
✅ **Vector-Based Semantic Search** – Employs **ChromaDB & LangChain embeddings** for AI-powered data retrieval.  
✅ **SQL & NoSQL Support** – Executes dynamic **SQL queries for structured data** and **MongoDB queries for document-based data**.  
✅ **Intelligent Summarization & Visualization** – Generates AI-driven insights and suggests the best chart types.  
✅ **Session Management** – Saves conversation history for future reference.  
✅ **Error Handling & Logging** – Provides robust logging for debugging and monitoring.  

---

## **🛠️ Tech Stack**
- **Frontend**: Streamlit (for UI)
- **Databases**: SQLite, MongoDB, ChromaDB (Vector DB)
- **AI & NLP**: Google Gemini AI, LangChain, Hugging Face Embeddings
- **Data Processing**: Pandas, NumPy, Matplotlib
- **Backend Logic**: Python, Dynamic Query Execution
- **Security & Config Management**: dotenv (Environment Variables)

---

## **🔧 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/avsai24/CaptenAI_Query_Genius.git
cd CaptenAI-Query-Genius
```

### **2️⃣ Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\\Scripts\\activate     # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Set Up Environment Variables**
Create a `.env` file and add your API keys and database paths:
```
API_KEY=your-google-gemini-api-key
DATABASE_FOLDER=path-to-your-sqlite-databases
MONGO_URL=your-mongodb-url
```

### **5️⃣ Run the Application**
```bash
streamlit run app.py
```

---

## **⚙️ How It Works**
1️⃣ **User Inputs a Query** → AI decides whether to fetch data from **SQLite, MongoDB, or VectorDB**.  
2️⃣ **Query Execution** → Runs **SQL queries for structured data**, **MongoDB queries for NoSQL data**, or **Vector similarity searches**.  
3️⃣ **Data Processing** → If multiple sources return results, they are **merged intelligently**.  
4️⃣ **Summarization & Visualization** → AI generates concise **summaries** and **auto-suggests the best chart type**.  
5️⃣ **Response Display** → The response is displayed, logged, and saved in the **conversation history**.  

---

## **📊 Use Cases**
- 📊 **Business Analytics** – Fetch sales reports, employee salaries, and revenue insights.  
- ⚽ **Sports Data** – Retrieve player statistics, team rankings, and match records.  
- 🏡 **Real Estate** – Analyze property prices, land zoning, and utility information.  
- 🤖 **AI-Powered Search** – Perform **semantic search** on AI-related datasets.  
- 🍽️ **E-commerce & Reviews** – Query Airbnb listings, movie ratings, restaurant reviews, etc.  

---

## **📸 Screenshots**
> User Interface.
![User Interface](https://github.com/avsai24/CaptenAI_Query_Genius/blob/main/images/first_image.png)

>When User Asks Question.
![User Question](https://github.com/avsai24/CaptenAI_Query_Genius/blob/main/images/second_image.png)

>You Can Also Delete All The History.
![Deleting History](https://github.com/avsai24/CaptenAI_Query_Genius/blob/main/images/third_image.png)
---

## **📜 Contribution Guidelines**
We welcome contributions! 🚀 To contribute:  
1. **Fork the repository**  
2. **Create a new branch** (`git checkout -b feature-new-functionality`)  
3. **Commit your changes** (`git commit -m "Added new feature"`)  
4. **Push to GitHub** (`git push origin feature-new-functionality`)  
5. **Create a Pull Request** 🎉  

---

## **📄 License**
This project is licensed under the **MIT License**. Feel free to use and modify it.  

---

## **📬 Contact**
📧 Email: venkatasaiancha24@gmail.com  
💻 LinkedIn: https://www.linkedin.com/in/venkatasaiancha/  
📂 GitHub: https://github.com/avsai24  

---

### **🌟 If you like this project, give it a ⭐ on GitHub!**
