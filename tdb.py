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

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_Key")

# Set device for PyTorch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define the folder path where PDF files are located
pdf_folder_path = './dateset1'

# List PDF files in the folder
os.listdir(pdf_folder_path)

# Load PDF files using UnstructuredPDFLoader
loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]

# Initialize an empty list to store extracted dates
date = []

# Iterate through loaders to load PDFs and extract dates
for loader in loaders:
    date.extend(loader.load())

# Login to Hugging Face model repository
login(token=os.getenv("Huggingface"))

# Initialize HuggingFace embeddings
embeddings = HuggingFaceInstructEmbeddings(cache_folder="./embeddings", model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE})

# Initialize text splitter for breaking down documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)

# Split documents into chunks
documents = text_splitter.split_documents(date)

# Create Chroma index from documents
db = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

# Create Chroma instance using persisted data
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Perform similarity search
docs = db3.similarity_search("Occupational Safety and Health Admin")

# Print most similar document
print(docs[0])
