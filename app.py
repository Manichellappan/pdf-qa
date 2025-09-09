import os
from pathlib import Path

# --- Robust fix for Spaces: force a writable config dir ---
# Prefer /app (writable in Spaces). If /app is not writable for some reason,
# fall back to /tmp so we never try to write to '/'.
preferred_home = Path("/app")
fallback_home = Path("/tmp")

try:
    # Force HOME to preferred writable location (override any existing value)
    os.environ["HOME"] = str(preferred_home)
    preferred_home.mkdir(parents=True, exist_ok=True)
    config_dir = preferred_home / ".streamlit"
    config_dir.mkdir(parents=True, exist_ok=True)
except PermissionError:
    # If /app isn't writable, use /tmp
    os.environ["HOME"] = str(fallback_home)
    config_dir = fallback_home / ".streamlit"
    config_dir.mkdir(parents=True, exist_ok=True)

# Also explicitly tell Streamlit where to put config files
os.environ["STREAMLIT_CONFIG_DIR"] = str(config_dir)

import streamlit as st
import tempfile
import logging
import shutil
from ingest import ingest_large_pdf
from query_engine import load_query_engine, ask_question

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# --- Safe writable directories (works with Dockerfile changes) ---
# Prefer a relative path inside the WORKDIR (/app). If that fails, fallback to tmp dir.
try:
    INDEX_DIR = os.path.join(os.getcwd(), "indexes")
    os.makedirs(INDEX_DIR, exist_ok=True)
    logging.info(f"Using INDEX_DIR: {INDEX_DIR}")
except PermissionError:
    logging.warning("PermissionError creating /app/indexes; falling back to a temporary directory")
    INDEX_DIR = tempfile.mkdtemp(prefix="indexes_")
    logging.info(f"Falling back to INDEX_DIR: {INDEX_DIR}")

# Ensure Streamlit home is set to a writable dir (Dockerfile sets this to /tmp/.streamlit)
if "STREAMLIT_HOME" not in os.environ:
    os.environ["STREAMLIT_HOME"] = "/tmp/.streamlit"
try:
    os.makedirs(os.environ["STREAMLIT_HOME"], exist_ok=True)
except Exception as e:
    logging.warning(f"Could not create STREAMLIT_HOME directory: {e}")

st.set_page_config(page_title="PDF Q&A", page_icon="📘", layout="wide")
st.title("📘 PDF Q&A with Local LLM (HuggingFace Space)")

# --- Sidebar: Upload PDF and manage FAISS Index ---
st.sidebar.header("PDF & FAISS Index Management")

# Upload new PDF
uploaded_pdf = st.sidebar.file_uploader("Upload PDF to create index", type=["pdf"], key="pdf_upload")
if uploaded_pdf:
    # Save uploaded PDF inside the app working dir to ensure writable path
    pdf_path = os.path.join(os.getcwd(), "uploaded.pdf")
    try:
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.sidebar.success("✅ PDF uploaded")
    except Exception as e:
        st.sidebar.error(f"Failed to save uploaded PDF: {e}")
        logging.exception(e)
        pdf_path = None

    # Ask for index name
    if pdf_path:
        index_name = st.sidebar.text_input("Enter index name for this PDF:",
                                          value=f"index_{len([d for d in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, d))]) + 1}")

        if st.sidebar.button("🔍 Build FAISS Index"):
            if not index_name.strip():
                st.sidebar.warning("Please provide a valid index name")
            else:
                index_path = os.path.join(INDEX_DIR, index_name)
                try:
                    with st.spinner("Processing PDF and building FAISS index... ⏳"):
                        os.makedirs(index_path, exist_ok=True)
                        ingest_large_pdf(pdf_path, index_path=index_path)
                    st.sidebar.success(f"🎉 FAISS index created: {index_name}")
                except Exception as e:
                    st.sidebar.error(f"Failed to build FAISS index: {e}")
                    logging.exception(e)

                # Download button
                if os.path.isdir(index_path) and st.sidebar.button(f"⬇️ Download '{index_name}'"):
                    try:
                        shutil.make_archive(index_path, 'zip', index_path)
                        with open(f"{index_path}.zip", "rb") as f:
                            st.sidebar.download_button(
                                label=f"Download FAISS Index '{index_name}'",
                                data=f,
                                file_name=f"{index_name}.zip",
                                mime="application/zip"
                            )
                    except Exception as e:
                        st.sidebar.error(f"Failed to create/download archive: {e}")
                        logging.exception(e)

# Upload previously saved FAISS index (.zip)
st.sidebar.subheader("Or Upload Existing FAISS Index")
uploaded_index = st.sidebar.file_uploader("Upload FAISS Index (.zip)", type=["zip"], key="index_upload")
if uploaded_index:
    save_name = st.sidebar.text_input("Enter name to save this index:",
                                      value=f"uploaded_index_{len([d for d in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, d))]) + 1}")
    if st.sidebar.button("📂 Save Uploaded Index"):
        save_path = os.path.join(INDEX_DIR, save_name)
        try:
            os.makedirs(save_path, exist_ok=True)
            temp_zip = os.path.join(save_path, "temp.zip")
            with open(temp_zip, "wb") as f:
                f.write(uploaded_index.getbuffer())
            shutil.unpack_archive(temp_zip, save_path)
            os.remove(temp_zip)
            st.sidebar.success(f"✅ FAISS index saved as {save_name}")
        except Exception as e:
            st.sidebar.error(f"Failed to save uploaded index: {e}")
            logging.exception(e)

# --- Main Q&A Section ---
# List all session FAISS indexes (directories inside INDEX_DIR)
try:
    available_indexes = [f for f in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, f))]
except Exception as e:
    available_indexes = []
    logging.exception(e)

if available_indexes:
    st.subheader("Ask questions about your document(s)")
    selected_index = st.selectbox("Select FAISS Index:", available_indexes)
    try:
        qa = load_query_engine(os.path.join(INDEX_DIR, selected_index))
    except Exception as e:
        st.error(f"Failed to load selected index: {e}")
        qa = None
        logging.exception(e)

    query = st.text_input("💡 Enter your question:")
    if query and qa:
        with st.spinner("Thinking... 🤔"):
            try:
                answer = ask_question(query, qa)
                st.write("### 📖 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error during QA: {e}")
                logging.exception(e)
else:
    st.info("👈 Upload a PDF or FAISS index to start querying")
