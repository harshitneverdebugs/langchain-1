import google.generativeai as genai
import os
import streamlit as st
import faiss
import os
import pickle
import time
import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

api_key="AIzaSyDn_GMDDnrTMLZ79YlVisg_DvFKIBiQdi8"

st.title("News research Tool📈")
st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Process URL")
# Defining file paths
index_file_path = "vector_index.faiss"
meta_file_path = "vector_index_meta.pkl"

main_placeholder=st.empty()
vectorstore = None

if process_url_clicked:
    if not urls:
        main_placeholder.error("Please enter at least one URL.")
    else:
        try:
            # Loading the data
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data Loading... Started... ✅")
            data = loader.load()

            # Splitting the data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitting... Started... ✅")
            docs = text_splitter.split_documents(data)

            # Creating embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            vectorindex_genai = FAISS.from_documents(docs, embeddings)

            # Save FAISS index
            faiss.write_index(vectorindex_genai.index, index_file_path)

            # Save metadata separately
            with open(meta_file_path, "wb") as f:
                pickle.dump(vectorindex_genai.docstore._dict, f)

            main_placeholder.text("Embedding Vector Built Successfully ✅")
        except Exception as e:
            main_placeholder.error(f"Error: {e}")

# Load metadata and vectorstore if files exist
if os.path.exists(index_file_path) and os.path.exists(meta_file_path):
    try:
        faiss_index = faiss.read_index(index_file_path)
        with open(meta_file_path, "rb") as f:
            metadata = pickle.load(f)
        vectorstore = FAISS(embeddings.embed_query, faiss_index, metadata)
    except Exception as e:
        main_placeholder.error(f"Error loading vectorstore: {e}")

query = main_placeholder.text_input("Question:")

if query and vectorstore:
    try:
        llm = ChatGoogleGenerativeAI(
            max_tokens=500,
            model='gemini-1.5-pro',
            google_api_key=api_key
        )
        retriever = vectorstore.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
        result = chain({"question": query}, return_only_outputs=True)

        # Display the answer
        st.header("Answer")
        st.write(result["answer"])

        # Display the sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  
            for source in sources_list:
                st.write(source)
    except Exception as e:
        st.error(f"Error during query processing: {e}")
