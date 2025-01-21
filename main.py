from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

import streamlit as st

load_dotenv()

def setup_qa_system(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()

    llm = ChatOpenAI(model="gpt-4o-mini")
    qa_chain = RetrievalQA.from_chain_type(retriever=retriever, llm=llm)

    return qa_chain

if __name__ == "__main__":
    st.title("Document QA System")

    file_path = 'documents/Income_Tax_law_of_2022.pdf'
    if file_path:
        qa_chain = setup_qa_system(file_path)
        question = st.text_input("Enter a question")
        if question:
            response = qa_chain.invoke(question)
            st.write(response['result'])