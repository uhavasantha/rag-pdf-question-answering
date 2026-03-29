import streamlit as st
from utils.rag import process_docs, ask_question

st.title("📄 Knowledge Base Search Engine")

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    with open("temp.pdf", "wb") as f:
        f.write(file.read())

    db = process_docs("temp.pdf")

    query = st.text_input("Ask a question")

    if query:
        answer = ask_question(db, query)
        st.write(answer)