import os
import warnings
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import torch
import requests
from typing import List
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.base import Embeddings

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
warnings.filterwarnings("ignore")

# Define Flask app
app = Flask(__name__)
CORS(app)


llm = ChatOpenAI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="FAKE",
    max_tokens=512,
    temperature=0.7
)

def get_embedding(text: str) -> List[float]:
    response = requests.post("http://0.0.0.0:5002/embeddings",
        json={"model": "BAAI/bge-small-en-v1.5", "input": [text]})
    if response.status_code == 200:
        data = response.json()
        return data['data'][0]['embedding']
    else:
        raise Exception(f"API request failed with status code {response.status_code}")

class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return get_embedding(text)

def process_text(text):
    text = "\n".join([doc.page_content for doc in text])
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=32,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = CustomEmbeddings()
    
    knowledgeBase = FAISS.from_texts(chunks, embedding=embeddings)
    return knowledgeBase


def load_text_files_from_directory(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path)
            all_docs.extend(loader.load())
    return all_docs

@app.route('/process_pdf', methods=['POST', 'GET'])
def process_pdf():
    try:
        tcno = request.headers.get('tcno')
        print("tcno::::::",tcno)
        if not tcno:
            return jsonify({'error': 'Missing tcno header'}), 400
        
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        folder_path = f"/data/tendergpt/livetender_txt/{tcno}"
        all_docs = load_text_files_from_directory(folder_path)
        knowledge_base = process_text(all_docs)
        torch.cuda.empty_cache()
        # Process single query
        result = process_query(query, knowledge_base)
        
        # Clear CUDA cache after processing the query
        torch.cuda.empty_cache()
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500

        return jsonify({'response': result['response']})

    except Exception as e:
        return jsonify({'error': 'Failed to process PDF', 'details': str(e)}), 500

def process_query(query, knowledge_base):
    try:
        docs = knowledge_base.similarity_search(query)
        chain = load_qa_chain(llm, chain_type='stuff')
        with get_openai_callback() as cost:
            response = chain.run(input_documents=docs, question=query)
            print(cost)
        return {'response': response}
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
