# Install dependencies if not already done:
# !pip install -U langchain-google-genai chromadb langchain tiktoken

import os
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Set your Google Gemini API key as environment variable or enter manually
import os
api_key = "Enter api key"
#mask it at a later stage
os.environ["GOOGLE_API_KEY"] = api_key

# Step 1: Load your document as a string
# from DocUpPa import pdf_content
# document_string = pdf_content
document_string = "Your string"

# Step 2: Convert document string to LangChain Document
doc = Document(page_content=document_string, metadata={"source": "local_document"})

# Step 3: Split document into manageable chunks (adjust chunk_size as needed)
splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents([doc])

# Step 4: Embed the document chunks using the Google Gemini Embeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Step 5: Store embeddings in Chroma vector store
vectorstore = Chroma.from_documents(docs, embeddings)

# Step 6: Create a retriever from the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 7: Initialize the Google Gemini chat model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Step 8: Create a RetrievalQA chain with the model and retriever
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Step 9: Interactive Q&A loop
while True:
    question = input("Enter your question (or type 'exit' to quit): ")
    if question.lower() == "exit":
        break
    answer = qa_chain.run(question)
    print("Answer:", answer)
