import streamlit as st
import os
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)  # Make sure it prints the correct key

st.set_page_config(page_title="ALI AI PRO", page_icon="🤖", layout="wide")

st.title("🤖 ALI AI PRO - Advanced Knowledge Assistant")

if not api_key:
    st.error("❌ OPENAI_API_KEY missing in .env file")
    st.stop()

st.success("✅ API Key Loaded Successfully")

uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")

if uploaded_file:

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI

    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # LLM
    llm = ChatOpenAI(temperature=0)

    question = st.text_input("💬 Ask a question from your PDF")

    if question:
        retriever = vectorstore.as_retriever()
        relevant_docs = retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        Answer the question based only on the context below:

        {context}

        Question: {question}
        """

        response = llm.invoke(prompt)

        st.success(response.content)
