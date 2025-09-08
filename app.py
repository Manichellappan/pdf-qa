import streamlit as st
import os
from ingest import ingest_large_pdf
from query_engine import load_query_engine, ask_question
import shutil

# Folder to store session FAISS indexes
INDEX_DIR = "indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

st.set_page_config(page_title="PDF Q&A", page_icon="📘", layout="wide")
st.title("📘 PDF Q&A with Local LLM (HuggingFace Space)")

# --- Sidebar: Upload PDF and manage FAISS Index ---
st.sidebar.header("PDF & FAISS Index Management")

# Upload new PDF
uploaded_pdf = st.sidebar.file_uploader("Upload PDF to create index", type=["pdf"], key="pdf_upload")
if uploaded_pdf:
    pdf_path = os.path.join("uploaded.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.sidebar.success("✅ PDF uploaded")

    # Ask for index name
    index_name = st.sidebar.text_input("Enter index name for this PDF:", value=f"index_{len(os.listdir(INDEX_DIR)) + 1}")

    if st.sidebar.button("🔍 Build FAISS Index"):
        if not index_name.strip():
            st.sidebar.warning("Please provide a valid index name")
        else:
            index_path = os.path.join(INDEX_DIR, index_name)
            with st.spinner("Processing PDF and building FAISS index... ⏳"):
                ingest_large_pdf(pdf_path, index_path=index_path)
            st.sidebar.success(f"🎉 FAISS index created: {index_name}")

            # Download button
            if st.sidebar.button(f"⬇️ Download '{index_name}'"):
                shutil.make_archive(index_path, 'zip', index_path)
                with open(f"{index_path}.zip", "rb") as f:
                    st.sidebar.download_button(
                        label=f"Download FAISS Index '{index_name}'",
                        data=f,
                        file_name=f"{index_name}.zip",
                        mime="application/zip"
                    )

# Upload previously saved FAISS index (.zip)
st.sidebar.subheader("Or Upload Existing FAISS Index")
uploaded_index = st.sidebar.file_uploader("Upload FAISS Index (.zip)", type=["zip"], key="index_upload")
if uploaded_index:
    save_name = st.sidebar.text_input("Enter name to save this index:", value=f"uploaded_index_{len(os.listdir(INDEX_DIR)) + 1}")
    if st.sidebar.button("📂 Save Uploaded Index"):
        save_path = os.path.join(INDEX_DIR, save_name)
        os.makedirs(save_path, exist_ok=True)
        temp_zip = os.path.join(save_path, "temp.zip")
        with open(temp_zip, "wb") as f:
            f.write(uploaded_index.getbuffer())
        shutil.unpack_archive(temp_zip, save_path)
        os.remove(temp_zip)
        st.sidebar.success(f"✅ FAISS index saved as {save_name}")

# --- Main Q&A Section ---
# List all session FAISS indexes
available_indexes = [f for f in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, f))]
if available_indexes:
    st.subheader("Ask questions about your document(s)")
    selected_index = st.selectbox("Select FAISS Index:", available_indexes)
    qa = load_query_engine(os.path.join(INDEX_DIR, selected_index))

    query = st.text_input("💡 Enter your question:")
    if query:
        with st.spinner("Thinking... 🤔"):
            answer = ask_question(query, qa)
        st.write("### 📖 Answer")
        st.write(answer)
else:
    st.info("👈 Upload a PDF or FAISS index to start querying")
