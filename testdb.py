import cv2
import openai
from langchain.vectorstores import Chroma
import os
from flask import Flask, request, jsonify, render_template, Response, make_response
from flask_cors import CORS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
embeddings = OpenAIEmbeddings(openai_api_key="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")
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db3.as_retriever(search_kwargs={"k": 2})
retrieved_docs = retriever.get_relevant_documents(
    "What are the approaches to Task Decomposition?"
)
print(retrieved_docs)