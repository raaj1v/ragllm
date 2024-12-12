
#############opensearch datavbase elastice #####


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
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from opensearchpy import OpenSearch
from elasticsearch import Elasticsearch
# import eventlet
# eventlet.monkey_patch()

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Elasticsearch client setup
es_client = Elasticsearch(['http://10.0.0.200:9200'])

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
    hosts=['https://10.0.0.109:9200'],
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
# Initialize SocketIO with eventlet for async support
# socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# Initialize embedding model globally
embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# Load LLM model globally
llm = ChatOpenAI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="FAKE",
    max_tokens=2048,
    temperature=0.7
)

# Text processing functions
def process_text(text):
    text_content = "\n".join([doc.page_content for doc in text])
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2048,
        chunk_overlap=32,
        length_function=len
    )
    chunks = text_splitter.split_text(text_content)
    knowledge_base = FAISS.from_texts(chunks, embedding_model)
    return knowledge_base

def load_text_files_from_directory(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path)
            all_docs.extend(loader.load())
    return all_docs

query_titles ={
    "Please extract and summarize the following information from the attached tender document which includes Identification and summarization of any clauses that specify the prequalification requirements or eligibility criteria for bidders. Highlight the technical requirements, Performance criteria, financial criteria and specifications that bidders must meet and Summarize any non-functional criteria mentioned in the document, such as performance, security, usability, or compliance requirements. Please provide Page No. and File name from which details is extracted and summarized": "Eligibility Criteria",
    "What are the functional requirements, also known as the scope of work, mentioned in the document?": "Scope of Work"
}

predefined_queries = list(query_titles.keys())

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
    print('Received process_pdf event with tc_no:',tc_no)
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
        socketio.emit('pdf_processed_complete', json.dumps({'status': 'complete','tabId':tabId}))
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
        all_docs = load_text_files_from_directory(folder_path)
        knowledge_base = process_text(all_docs)
        print("Knowledge base created successfully")

        for query in predefined_queries:
            try:
                result = process_query(query, knowledge_base, tabId)
                results.append(result)
                yield result
            except Exception as e:
                result = {'title': query_titles[query], 'error': str(e)}
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
        # print("Connection successful")

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

def process_query(query, knowledge_base, tabId):
    try:
        docs = knowledge_base.similarity_search(query)
        chain = load_qa_chain(llm, chain_type='stuff')
        with get_openai_callback() as cost:
            response = chain.run(input_documents=docs, question=query)
        return {'title': query_titles[query], 'response': response, 'tabId': tabId}
    except Exception as e:
        return {'title': query_titles[query], 'error': str(e), 'tabId': tabId}

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5003)
