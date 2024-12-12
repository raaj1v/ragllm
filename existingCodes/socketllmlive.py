import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore", category=UserWarning,message=".*LangChainDeprecationWarning.*")
from langchain.chains.question_answering import load_qa_chain
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
import json
from llama_index.core.base.response.schema import Response
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from flask import Flask, Response, request, jsonify
from langchain_community.document_loaders import TextLoader
import torch

app = Flask(__name__)
CORS(app)

print(os.urandom(24))
app = Flask(__name__)
app.config['SECRET_KEY'] =os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")  
 
# # Initialize the embedding model globally
embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
   
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

query_titles = {
    "What are the functional requirements, also known as the scope of work, mentioned in the document?": "Scope of Work",
    "Please extract and summarize the following information from the attached tender document which includes Identification and summarization of any clauses that specify the prequalification requirements or eligibility criteria for bidders / Highlight the technical requirements or specifications that bidders must meet and Summarize any non-functional criteria mentioned in the document, such as performance, security, usability, or compliance requirements.": "Eligibility Criteria",
}

predefined_queries = list(query_titles.keys())

@socketio.on('message')
def handle_message(message):
    socketio.send(message, broadcast=True)

@socketio.on('process_pdf')
def handle_process_pdf(tc_no):
    print('Received process_pdf event with tc_no:', tc_no)
    try:
        for response in process_pdf(tc_no):
            try:
                response_json = json.dumps(response, ensure_ascii=False)
                print("streaming started")
                socketio.emit('response', response_json)
                print('Emission successful:', response_json)
            except Exception as e:
                print('Emission error:', e)
        socketio.emit('pdf_processed_complete', json.dumps({'status': 'complete'}))
    except Exception as e:
        print('Error during PDF processing:', str(e))
        socketio.emit('pdf_processed_error', json.dumps({'error': str(e)}))

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def process_pdf(tcno):
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
                result = process_query(query, knowledge_base)
                yield result
            except Exception as e:
                yield {'title': query_titles[query], 'error': str(e)}

    except Exception as e:
        yield {'error': 'Failed to process PDF', 'details': str(e)}

def process_query(query, knowledge_base):
    try:
        docs = knowledge_base.similarity_search(query)
        llm = ChatOpenAI(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            openai_api_base="http://localhost:8000/v1",
            openai_api_key="FAKE",
            max_tokens=2048,
            temperature=0.7
        )
        chain = load_qa_chain(llm, chain_type='stuff')
        with get_openai_callback() as cost:
            response = chain.run(input_documents=docs, question=query)
        return {'title': query_titles[query], 'response': response}
    except Exception as e:
        return {'title': query_titles[query], 'error': str(e)}

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5003)