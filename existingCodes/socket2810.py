

query_tittles = {
    "What are the functional requirements, also known as the scope of work, mentioned in the document?": "Scope of Work",
    "Clauses specifying  Pre-Qualification Criteria  or eligibility criteria": "Prequalification Criteria",
    "List all mandatory qualification criteria, including Blacklisting and required certifications": "Mandatory Qualification Criteria",
     "Performance criteria including work experience,experience and past performance criteria, emphasizing the need for prior similar project experience, references, and the successful completion of similar contracts": "Performance Criteria",
     "Financial criteria including turnover, Networth": "Financial Criteria",
    "Technical requirements": "Technical Requirements",
     "Work Specifications that bidders must meet to deliver tender requirements": "Specifications",
     "Supporting documents": "Supporting Documents",
#      "List of all the dates mentioned in the tender document which should include Bid submission end date or due date of tender, Bid validity, Opening date, closing date, pre bid meeting date, EMD date":"Importants Date",
     "Extract a comprehensive list of all dates, times, and monetary values, along with their specific labels or descriptions as mentioned in the document. This includes but is not limited to the following fields: bid submission end date, tender due date, bid validity, opening date, closing date, pre-bid meeting date, EMD date, tender value, and tender fee. Group all extracted items under the label 'Important Dates and Amounts,' clearly specifying each date, time, or amount and its description as stated in the document.":"Important date",
     "Extract the contact details, including phone number, email address, and officer name. If the details are unavailable, return 'None' for the missing fields.":"Contact details"
       
}


#############opensearch datavbase elastic#####

import os
import json
import pyodbc
import warnings
import torch
from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
# from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from opensearchpy import OpenSearch
from elasticsearch import Elasticsearch
import requests
from typing import List
from langchain.embeddings.base import Embeddings
import sys, ast
sys.path.append(r'/data/QAAPI')
sys.path.append(r'/data/QAAPI/qaVenv2')
sys.path.append(r'/pharma/')
sys.path.append(r'/pharma/MEvenv/')
# import API_regen_fetch_BOQ as boqapi

from flask import Flask, jsonify, request  # Add `request` here


# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TORCH_USE_CUDA_DSA'] = "0"
# Elasticsearch client setup
es_client = Elasticsearch(['http://141.148.218.254:9200'])

def update_elasticsearch(tcno):
    search_query = {"query": {"term": {"tcno": tcno}}}
    indices = ['tbl_tendersadvance_migration_embedding', 'tbl_tendersadvance_migration']

    for index in indices:
        es_response = es_client.search(index=index, body=search_query)
        if es_response['hits']['total']['value'] > 0:
            for hit in es_response['hits']['hits']:
                es_client.update(index=index, id=hit['_id'], body={"doc": {"ispq": 1}})
                print(f"Updated tcno {tcno} with ispq = 1 in {index}")

# OpenSearch client setup
index_name = 'tprocanswers'
client = OpenSearch(
    hosts=['https://localhost:9200'],
    http_auth=("admin", "4Z*lwtz,,2T:0TGu"),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False
)

# Create index if it doesn't exist
if not client.indices.exists(index=index_name):
    client.indices.create(index=index_name)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*")

# Load LLM model globally
llm = ChatOpenAI(
    # model_name="meta-llama/Meta-Llama-3-8B-Instruct",
     model_name="meta-llama/Llama-3.1-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="FAKE",
    max_tokens=4096,
    temperature=0.1
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

def process_text(texts):
    # Join all document texts into a single string to preserve context
    combined_text = "\n".join(texts)
    
    # Text splitting based on semantic boundaries to keep context intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=32,
        length_function=len,
        separators=["\n\n", "\n", ".", " "]
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(combined_text)
    
    # Create embeddings for the knowledge base
    embeddings = CustomEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embedding=embeddings)
    return knowledge_base

def load_text_files_from_directory(folder_path):
    all_text = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path)
            docs = loader.load()
            # Extract text content from Document objects
            all_text.extend(doc.page_content for doc in docs if hasattr(doc, 'page_content'))
    return all_text

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_established', {'data': 'Connected to server'})

# SocketIO event handlers
@socketio.on('message')
def handle_message(message):
    print('Received message:', message)
    socketio.send(message)

@socketio.on('process_pdf')
def handle_process_pdf(tc_no, tabId):
    print('Received process_pdf event with tc_no:', tc_no)
    try:
        results = []
        for response in process_pdf(tc_no, tabId, results):
            try:
                response_json = json.dumps(response, ensure_ascii=False)
                print("Streaming started")
                socketio.emit('response', response_json)
                print('Emission successful:', response_json)
            except Exception as e:
                print('Emission error:', e)
        socketio.emit('pdf_processed_complete', json.dumps({'status': 'complete', 'tabId': tabId}))
    except Exception as e:
        print('Error during PDF processing:', str(e))
        socketio.emit('pdf_processed_error', json.dumps({'error': str(e)}))

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def process_pdf(tcno, tabId, results):
    try:
        print("Processing PDF for tcno:", tcno)
        if not tcno:
            yield {'error': 'Missing tcno header'}
            return

        folder_path = f"/data/tendergpt/livetender_txt/{tcno}"
        all_docs_text = load_text_files_from_directory(folder_path)
        knowledge_base = process_text(all_docs_text)
        print("Knowledge base created successfully")

        for query, title in query_tittles.items():
            try:
                result = process_query(query, title, knowledge_base, tabId)
                results.append(result)
                yield result
                torch.cuda.empty_cache()
            except Exception as e:
                result = {'title': title, 'error': str(e)}
                results.append(result)
                yield result

        # Index results into OpenSearch
        doc_id = tcno  # Use tcno as the document ID
        json_response = {"results": results}
        response = client.index(index=index_name, id=doc_id, body=json_response)
        print(f"Indexed results for {tcno} in OpenSearch: {response}")

        # Call the function to update Elasticsearch indices
        update_elasticsearch(tcno)

        # Database connection parameters
        server = '10.0.0.63'
        database = 'ttneo'
        username = 'aimlpq'
        password = 'aimlpq'
        driver = '/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.3.so.3.1'

        # Connection string
        conn_string = (
            f'DRIVER={driver};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password};'
            'TrustServerCertificate=yes;'
        )

        # Establishing the connection
        conn = pyodbc.connect(conn_string)

        # Update the `ispq` column in the table based on the `tcno`
        cursor = conn.cursor()
        update_query = "UPDATE apptender.tbl_tender SET ispq = 1 WHERE tcno = ?"
        cursor.execute(update_query, (tcno,))

        # Commit the transaction
        conn.commit()
        print("Database updated successfully")

        # Close the cursor and connection
        cursor.close()
        conn.close()

        # Clear CUDA cache
        torch.cuda.empty_cache()

    except Exception as e:
        yield {'error': 'Failed to process PDF', 'details': str(e)}

def process_query(query, title, knowledge_base, tabId):
    try:
        docs = knowledge_base.similarity_search(query)
        chain = load_qa_chain(llm, chain_type='stuff')
        with get_openai_callback() as cost:
            response = chain.run(input_documents=docs, question=query)
        return {'title': title, 'response': response, 'tabId': tabId}
    except Exception as e:
        return {'title': title, 'error': str(e), 'tabId': tabId}

# @app.route('/process_files', methods=['POST'])
# def process_files1():
#     data = request.json
#     folder_path = data.get("folder_path")
#     tc_no = data.get("tc_no")
#     boq=  boqapi.process_files(folder_path,tc_no)
#     return boq


if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5003)
