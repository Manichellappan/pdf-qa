# ingest.py
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter
import tempfile
import os
import re
from PyPDF2 import PdfReader, PdfWriter

# === CONFIG ===
BATCH_SIZE = 50
MAX_TOKENS = 512
TOKEN_OVERLAP = 50

CAPTION_RE = re.compile(
    r'^\s*(fig(?:ure)?\.?\s*\d+[:\)\. -]*|table\s*\d+[:\)\. -]*|plate\s*\d+[:\)\. -]*|page\s*\d+(\s*of\s*\d+)?)',
    re.I | re.M
)

def clean_text(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        if CAPTION_RE.match(ln.strip()):
            continue
        if len(ln.strip()) <= 3 and re.fullmatch(r'[\W\d]+', ln.strip()):
            continue
        lines.append(ln)
    out = "\n".join(lines)
    return re.sub(r'\s+', ' ', out).strip()

def split_pdf(input_pdf, start, end):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    for i in range(start, min(end, len(reader.pages))):
        writer.add_page(reader.pages[i])
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    with open(temp_file.name, "wb") as f:
        writer.write(f)
    return temp_file.name

def ingest_large_pdf(pdf_path, index_path="faiss_index"):
    """Main function to build FAISS index from a PDF."""
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"✅ PDF has {total_pages} pages")

    vectorstore = None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    for start in range(0, total_pages, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_pages)
        print(f"Processing pages {start}–{end}...")

        batch_file = split_pdf(pdf_path, start, end)
        loader = UnstructuredPDFLoader(batch_file, strategy="ocr_only", mode="elements")
        docs = loader.load()
        os.remove(batch_file)

        splitter = TokenTextSplitter(chunk_size=MAX_TOKENS, chunk_overlap=TOKEN_OVERLAP)
        chunks = splitter.split_documents(docs)

        cleaned_chunks = []
        for i, doc in enumerate(chunks):
            text = clean_text(doc.page_content)
            if len(text.split()) < 20:
                continue
            doc.page_content = text
            doc.metadata.update({"page_range": f"{start}-{end}", "chunk_id": i})
            cleaned_chunks.append(doc)

        if not cleaned_chunks:
            continue

        if vectorstore is None:
            vectorstore = FAISS.from_documents(cleaned_chunks, embeddings)
        else:
            vectorstore.add_documents(cleaned_chunks)

    if vectorstore is not None:
        vectorstore.save_local(index_path)
        print(f"🎉 Index saved at {index_path}")
    else:
        print("⚠️ No chunks found. Index not created.")

# Optional CLI for testing
if __name__ == "__main__":
    pdf_path = "sample.pdf"
    ingest_large_pdf(pdf_path, index_path="faiss_index")
