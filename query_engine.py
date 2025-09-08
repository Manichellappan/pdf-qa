# query_engine.py
import os
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

MODEL_NAME = "google/flan-t5-large"
MAX_TOKENS = 300
TEMPERATURE = 0
MAX_CONTEXT_TOKENS = 500

def truncate_tokens(text, tokenizer, max_tokens=MAX_CONTEXT_TOKENS):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens)

def load_query_engine(index_path):
    """Load FAISS index and return a QA chain and tokenizer."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"{index_path} not found. Run ingest first.")

    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")

    local_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    llm = HuggingFacePipeline(pipeline=local_pipeline)

    prompt_template = """You are a reasoning assistant. Analyse and give the answer. 
Context:
{context}
Question:
{question}
Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    return {"qa_chain": qa_chain, "tokenizer": tokenizer}

def ask_question(query, qa_engine):
    """Ask a question against the PDF vectorstore with truncated context."""
    qa_chain = qa_engine["qa_chain"]
    tokenizer = qa_engine["tokenizer"]

    # Retrieve relevant docs
    retrieved_docs = qa_chain.retriever.invoke(query)
    full_context = " ".join([doc.page_content for doc in retrieved_docs])
    truncated_context = truncate_tokens(full_context, tokenizer)

    answer = qa_chain.invoke(truncated_context + "\n\nQuestion: " + query)
    return answer["result"]

# Optional CLI
if __name__ == "__main__":
    index_path = "faiss_index"  # default
    qa_engine = load_query_engine(index_path)
    while True:
        q = input("Ask: ")
        if q.lower() == "exit":
            break
        print(ask_question(q, qa_engine))
