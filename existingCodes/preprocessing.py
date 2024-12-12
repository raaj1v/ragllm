import os
# import aspose.words as aw
import shutil
from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings("ignore")
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain_community.vectorstores import LanceDB
from langchain.document_loaders import PyPDFLoader
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from itertools import chain
import patoolib
import requests
from llama_index.core import SimpleDirectoryReader, Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import time
import os
import patoolib
import requests
import base64
import io
import shutil
import os

def process_text(text):
    text = "\n".join([doc.page_content for doc in text])
    # text=""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2048,
        chunk_overlap=32,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

from langchain_community.document_loaders import TextLoader
def load_text_files_from_directory(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path)
            all_docs.extend(loader.load())
    return all_docs
