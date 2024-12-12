import os
import re
import warnings
import json
import numpy as np
import torch
from typing import Dict, List, Any
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from concurrent.futures import ThreadPoolExecutor, as_completed
from opensearchpy import OpenSearch

# Suppress warnings
warnings.filterwarnings("ignore")

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# OpenSearch client setup
index_name = 'tprocanswers'
client = OpenSearch(
    hosts=['https://localhost:9200'],
    http_auth=("admin", "4Z*lwtz,,2T:0TGu"),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False
)

class TenderAnalyzer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.llm = ChatOpenAI(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            openai_api_base="http://localhost:8000/v1",
            openai_api_key="FAKE",
            max_tokens=1024,
            temperature=0.1
        )
        self.chain = load_qa_chain(self.llm, chain_type='stuff')
        self.queries = {
            "What are the functional requirements, also known as the scope of work, mentioned in the document?": "Scope of Work",
            "Extract clauses that specify Pre-Qualification Criteria or eligibility criteria.": "Prequalification Criteria",
            "List all supporting documents required for this tender.": "Supporting Documents",
            "Extract a comprehensive list of all dates, times, and monetary values, along with their specific labels or descriptions as mentioned in the document.": "Important Dates",
            "Extract the contact details of the officer from this document, including their name, email ID, and contact number.": "Contact Details"
        }

    def process_document(self, file_path: str) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        sentences = self._split_into_sentences(text)
        chunks = self._create_chunks(sentences)
        return self._chunk_by_tokens(chunks)

    def _split_into_sentences(self, text: str) -> List[Dict[str, Any]]:
        sentences = [{'sentence': s, 'index': i} for i, s in enumerate(re.split(r'(?<=[.?!])\s+', text))]
        return self._combine_sentences(sentences)

    def _combine_sentences(self, sentences: List[Dict[str, Any]], buffer_size: int = 1) -> List[Dict[str, Any]]:
        combined = []
        for i, sent in enumerate(sentences):
            context = []
            for j in range(max(0, i - buffer_size), i):
                context.append(sentences[j]['sentence'])
            context.append(sent['sentence'])
            for j in range(i + 1, min(len(sentences), i + buffer_size + 1)):
                context.append(sentences[j]['sentence'])
            sent['combined_sentence'] = ' '.join(context)
            combined.append(sent)
        return combined

    def _create_chunks(self, sentences: List[Dict[str, Any]]) -> List[str]:
        embeddings = self.model.encode([s['combined_sentence'] for s in sentences])
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]))
            distances.append(1 - similarity)
        threshold = np.percentile(distances, 95)
        chunks = []
        start_idx = 0
        for i, distance in enumerate(distances):
            if distance > threshold:
                chunk = ' '.join([s['sentence'] for s in sentences[start_idx:i + 1]])
                chunks.append(chunk)
                start_idx = i + 1
        if start_idx < len(sentences):
            chunk = ' '.join([s['sentence'] for s in sentences[start_idx:]])
            chunks.append(chunk)
        return chunks

    def _chunk_by_tokens(self, texts: List[str], max_tokens: int = 3500) -> List[str]:
        max_chars = max_tokens * 2
        chunks = []
        for text in texts:
            text_chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
            chunks.extend(text_chunks)
        return chunks

    def process_query(self, query: str, text: str,tabId) -> str:
        try:
            with get_openai_callback() as cb:
                response = self.chain.run(
                    input_documents=[Document(page_content=text)],
                    question=query
                )
            return response.strip(),tabId
        except Exception as e:
            return f"Error: {str(e)}"

    def analyze_tender(self, file_path: str) -> Dict[str, str]:
        chunks = self.process_document(file_path)
        combined_text = " ".join(chunks)
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.queries)) as executor:
            future_to_query = {
                executor.submit(self.process_query, query, combined_text): title
                for query, title in self.queries.items()
            }
            for future in as_completed(future_to_query):
                title = future_to_query[future]
                try:
                    response = future.result()
                    results[title] = response
                except Exception as e:
                    results[title] = f"Error: {str(e)}"
        return results

# def process_pdf(tcno, tabId, results):
#     try:
#         print("Processing PDF for tcno:", tcno)
#         folder_path = f"/data/txtfolder/dailydocument_23-11-24_txt/{tcno}"
#         analyzer = TenderAnalyzer()
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 if file.endswith('.txt'):
#                     file_path = os.path.join(root, file)
#                     try:
#                         analysis = analyzer.analyze_tender(file_path)
#                         results.append(analysis)
#                         yield analysis
#                         torch.cuda.empty_cache()
#                     except Exception as e:
#                         yield {'error': str(e)}
#     except Exception as e:
#         yield {'error': str(e)}

from opensearchpy import OpenSearch

# Set up OpenSearch client
opensearch_client = OpenSearch(
    hosts=['https://localhost:9200'],
    http_auth=("admin", "4Z*lwtz,,2T:0TGu"),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False
)

index_name = 'tprocanswers'

# Function to index the results into OpenSearch
def index_results(tcno, results):
    try:
        opensearch_client.index(index=index_name, id=tcno, body={"results": results})
        print(f"Indexed results for {tcno} in OpenSearch.")
    except Exception as e:
        print(f"Error indexing results for {tcno}: {str(e)}")

# Modify the process_pdf function to index the results
def process_pdf(tcno, tabId, results):
    try:
        print("Processing PDF for tcno:", tcno)
        folder_path = f"/data/tendergpt/livetender_txt/{tcno}"
        analyzer = TenderAnalyzer()
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        analysis = analyzer.analyze_tender(file_path)
                        results.append(analysis)
                        yield analysis
                        # Index results in OpenSearch
                        index_results(tcno, analysis)
                        torch.cuda.empty_cache()
                    except Exception as e:
                        yield {'error': str(e)}
    except Exception as e:
        yield {'error': str(e)}


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_established', {'data': 'Connected to server'})

@socketio.on('process_pdf')
def handle_process_pdf(tc_no,tabId):
#     tc_no = data.get('tc_no')
#     tabId = data.get('tabId')
    
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

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5003)
