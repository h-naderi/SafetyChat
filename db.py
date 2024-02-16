import openai
import torch
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader,UnstructuredPDFLoader
import os
from langchain.embeddings.openai import OpenAIEmbeddings
openai.api_key = os.getenv("OPENAI_API_Key")
from huggingface_hub import login
from langchain.embeddings import HuggingFaceInstructEmbeddings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
pdf_folder_path = './date'
os.listdir(pdf_folder_path)
loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
date = []
for loader in loaders:
    date.extend(loader.load())
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(date)
embeddings = OpenAIEmbeddings(openai_api_key="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")
db = Chroma.from_documents(documents, embeddings,persist_directory="./chroma_db")