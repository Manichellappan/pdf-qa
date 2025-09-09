---
title: My Agentic AI Demo
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
---


PDF Q\&A System (Local LLM with FAISS Index)

Overview:
This project allows users to perform question-answering on their PDFs using a local LLM and FAISS vector database for embeddings. The application uses a Streamlit GUI for easy interaction.

Features:

* Upload PDFs and create FAISS indexes.
* Handle multiple PDFs in one session.
* Download FAISS indexes for later reuse.
* Ask questions from uploaded documents.
* Supports local inference using HuggingFace Flan-T5 model.

Folder Structure:

```
project_folder/
│
├── app.py                # Streamlit frontend
├── ingest.py             # Script to create FAISS index from PDFs
├── query_engine.py       # Script to load vectorstore and perform Q&A
├── indexes/              # Folder to store FAISS indexes (auto-created)
├── README.txt            # This file
└── requirements.txt      # Required Python packages
```

Getting Started (Local):

1. Clone the repository:

```
git clone <repo_url>
cd <repo_folder>
```

2. Create virtual environment and install dependencies:

```
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
pip install -r requirements.txt
```

3. Run the Streamlit app:

```
streamlit run app.py
```

4. Upload your PDF(s) and build FAISS index.
5. Optionally download the FAISS index to reuse later.
6. Ask questions in the main interface.

FAISS Index Management:

* Indexes are stored in `indexes/` folder.
* Users can create multiple indexes per session.
* Users can download any index as `.zip` and re-upload in future sessions.

HuggingFace Model:

* The app uses `google/flan-t5-large` from HuggingFace Hub.
* The model is downloaded automatically on first run.
* No need for users to download or install the model manually.

Deployment on HuggingFace Spaces:

* The Streamlit app can be hosted directly on HuggingFace Spaces.
* The `app.py`, `ingest.py`, `query_engine.py`, `requirements.txt` are needed.
* FAISS indexes can be downloaded locally; model is loaded on the server.
* Users interact via the web interface without needing to clone the repo.

Notes:

* The app supports handling multiple PDFs in one session.
* FAISS indexes persist across sessions if downloaded and re-uploaded.
* Ensure Python and required packages are correctly installed locally for testing.
* On HuggingFace Spaces, the model and environment are handled automatically.

Contact:
For any issues or contributions, please open an issue in the GitHub repository.
