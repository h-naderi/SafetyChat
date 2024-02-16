# Import necessary libraries
import openai
import torch
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader, UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from huggingface_hub import login
from langchain.embeddings import HuggingFaceInstructEmbeddings

openai.api_key = os.getenv("OPENAI_API_Key")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
pdf_folder_path = './date'
os.listdir(pdf_folder_path)

loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
date = []
for loader in loaders:
    date.extend(loader.load())

# Initialize text splitter for breaking down documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Split documents into chunks
documents = text_splitter.split_documents(date)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key="OPENAI_API_Key")

# Create Chroma index from documents
db = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
