<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=soft&color=0:0077b6,100:48cae4&height=180&section=header&text=RAG%20Chat%20App%20🚀&fontColor=ffffff&fontSize=50&animation=fadeIn" />
</p>

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
An intelligent, document-aware chatbot built with <b>Streamlit</b>, powered by <b>OpenAI</b>, <b>Pinecone</b>, and <b>MongoDB</b>. 
</p>

---

## 🌟 Key Features

- **🔐 Admin Authentication**
  - Default credentials: `admin` / `password 123`
  - Manage API keys, Mongo URI, Pinecone settings, and add new users.
  - All environment variables securely stored in MongoDB.

- **🧠 Conversational Memory**
  - Each user has personal chat sessions.
  - MongoDB-backed conversation history for continuity.
  - Intelligent recall of previous user questions.

- **📄 Document Ingestion**
  - Upload and embed `.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.log` files.
  - Documents are chunked, embedded (OpenAI), and stored in Pinecone.
  - Handles German umlauts gracefully (`ä→ae`, `ö→oe`, `ü→ue`, etc.).

- **⚙️ Dynamic Configuration**
  - Environment setup directly from the Streamlit UI.
  - Automatic app rerun after updates — no restart needed.

- **🧩 Modular Design**
  - Clean, scalable architecture — easy to extend or integrate.
  - Separate logic for UI, DB, RAG, utils, and configuration.

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

# 4. Run the app
streamlit run app.py
```

---

## ⚙️ Configuration

Log in as **admin** → open **Admin → Environment** to configure:

| Setting | Description |
|----------|--------------|
| **OpenAI API Key** | For embeddings & chat completions |
| **Pinecone API Key** | Pinecone project key |
| **Pinecone Host** | Example: `your-index.svc.region.pinecone.io` |
| **Mongo URI** | Example: `mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority` |
| **Mongo DB Name** | Default: `rag_chat` |

---

## 💡 Usage

### 👤 User Mode
- Chat with your uploaded documents (RAG).
- View your past conversations (MongoDB stored).

### 🛡️ Admin Mode
- Manage environment and user accounts.
- Upload or delete documents from Pinecone.
- Monitor ingestion status.

---

## 📄 Document Ingestion Workflow

1. Go to **Admin → Ingest**
2. Upload one or more supported documents.
3. Each file is:
   - Split into text chunks.
   - Embedded with OpenAI.
   - Stored in Pinecone along with metadata.
4. You can view or delete ingested documents anytime.

---

## 🧱 Project Structure

```
rag_chat_app/
├─ app.py
├─ requirements.txt
├─ config/
│  ├─ env.py
│  └─ paths.py
├─ db/
│  └─ mongo.py
├─ models/
│  └─ settings.py
├─ rag/
│  ├─ ingest.py
│  └─ pinecone_utils.py
├─ ui/
│  ├─ admin.py
│  └─ chat.py
└─ utils/
   ├─ chunk.py
   └─ ids.py
```

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **LLM & Embeddings** | [OpenAI API](https://platform.openai.com/) |
| **Vector DB** | [Pinecone](https://www.pinecone.io/) |
| **Database** | [MongoDB Atlas](https://www.mongodb.com/atlas) |
| **Validation** | [Pydantic v2](https://docs.pydantic.dev/) |
| **Language** | Python 3.10+ |

---

## 🌱 Future Enhancements

- OAuth / SSO (Google, Microsoft)
- Long-term chat summarization memory
- Namespace isolation per user
- Analytics dashboard (usage & embeddings stats)
- Notifications for ingestion status

---

## 🧰 Developer Tips

- Use `.env_settings.json` or the Admin UI for secret storage.
- Never commit `venv/` — include it in `.gitignore`.
- Use descriptive Pinecone namespaces for better organization.
- Test with `streamlit run app.py --server.enableCORS false` for local dev.

---

## 🪪 License

Released under the **MIT License** — free for personal and commercial use.

---