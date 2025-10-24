<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Pinecone-Vector_DB-1E90FF?logo=pinecone&logoColor=white" />
  <img src="https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

<h1 align="center">💬 RAG Chat — Streamlit + MongoDB + Pinecone</h1>

<p align="center">
An intelligent, document-aware chatbot powered by <b>OpenAI</b>, <b>Pinecone</b>, and <b>MongoDB</b> — built with <b>Streamlit</b> for a clean, modular UI.
</p>

---

## 🚀 Features

- **🔐 Admin Authentication**
  - Default credentials: `admin` / `password 123` (auto-seeded on first Mongo connection)
  - Add or remove users via the Admin Dashboard
  - Secure environment configuration stored in MongoDB

- **🧠 Conversational Memory**
  - Each user has an isolated chat workspace
  - Conversation history stored persistently in MongoDB
  - Smart memory window for coherent multi-turn chats

- **📄 Document Ingestion**
  - Upload and embed documents (`.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.log`)
  - Chunked and vectorized with OpenAI embeddings
  - Stored in Pinecone with umlaut-safe filenames and namespaces  
    (`ä→ae`, `ö→oe`, `ü→ue`, `Ä→Ae`, `Ö→Oe`, `Ü→Ue`, `ß→ss`)

- **⚙️ Admin Configuration UI**
  - Set all API keys (OpenAI, Pinecone, Mongo) directly in the interface
  - Automatic app rerun after saving new settings
  - Safe default environment loading — no KeyErrors

- **🧩 Modular Architecture**
  - Organized for clarity and extensibility: `ui/`, `db/`, `models/`, `rag/`, `utils/`, `config/`

---

## 🖥️ Installation

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
