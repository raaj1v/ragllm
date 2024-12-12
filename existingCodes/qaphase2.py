import sys, ast
sys.path.append(r'/data/imageExtraction/')
sys.path.append(r'/data/imageExtraction/imgEnv/')
import logging
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
from elasticsearch import Elasticsearch
from google.oauth2 import service_account
from googleapiclient.discovery import build
from apiclient.http import MediaFileUpload, MediaIoBaseDownload
from deep_translator import GoogleTranslator
from langdetect import detect
import io
import pandas as pd
from PyPDF2 import PdfReader
from PIL import Image
# from docx import Document
# from docx import Document as DocxDocument
from werkzeug.utils import secure_filename
import numpy as np
from flask import Flask, request, jsonify
import requests
# Suppress warnings
warnings.filterwarnings("ignore")

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/tender_uploads'
# Global variables to store uploaded document details and embeddings
uploaded_documents = {}
embeddings = {}

app.config['SECRET_KEY'] = os.urandom(24)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")



def get_embeddings_via_api(sentence):
    """Get embeddings from API (using all-mpnet-base-v2 model)"""
    response = requests.post(
        "http://0.0.0.0:5002/embeddings",
        json={"model": "sentence-transformers/all-MiniLM-L6-v2", "input": [sentence]}
    )
    return response.json()["data"][0]["embedding"]


# OpenSearch client setup
index_name = 'tprocanswers'
client = OpenSearch(
    hosts=['https://localhost:9200'],
    http_auth=("admin", "4Z*lwtz,,2T:0TGu"),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False
)

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


class TenderAnalyzer:
    def __init__(self):
#         self.model = SentenceTransformer(model_name)
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
            "List of all the dates mentioned in the tender document which should include Bid submission end date or due date of tender, Bid validity, Opening date, closing date, pre bid meeting date, EMD amount,tender fee, tender value": "Important Dates",
            "Extract the contact details of the officer from this document, including their name, email ID, and contact number.": "Contact Details"
        }
        self.max_chunk_tokens = 100000  # Safe limit below model's maximum
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
        """Create document chunks based on semantic similarity using API for embeddings"""
        embeddings = [get_embeddings_via_api(s['combined_sentence']) for s in sentences]
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
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



    def _estimate_tokens(self, text: str) -> int:
        """Estimate number of tokens in text (rough approximation)"""
        return len(text.split()) * 1.3  # Rough estimate of tokens
    
    def _chunk_by_tokens(self, texts: List[str]) -> List[str]:
        """Split texts into smaller chunks based on token count"""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for text in texts:
            estimated_tokens = self._estimate_tokens(text)
            
            if current_tokens + estimated_tokens > self.max_chunk_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [text]
                current_tokens = estimated_tokens
            else:
                current_chunk.append(text)
                current_tokens += estimated_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
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

    def analyze_tender(self, file_path: str, tabId: str) -> Dict[str, Any]:
        chunks = self.process_document(file_path)
        combined_text = " ".join(chunks)
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.queries)) as executor:
            future_to_query = {
                executor.submit(self.process_query, query, combined_text, tabId): title
                for query, title in self.queries.items()
            }
            for future in as_completed(future_to_query):
                title = future_to_query[future]
                try:
                    response, tabId = future.result()
                    results[title] = response
                except Exception as e:
                    results[title] = f"Error: {str(e)}"
        
        return {
            "tabId": tabId,
            "results": results
        }


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


def process_pdf(tcno, tabId, results):
    try:
        print("Processing PDF for tcno:", tcno)
        if not tcno:
            yield {'error': 'Missing tcno header'}
            return

        folder_path = f"/data/tendergpt/livetender_txt/{tcno}"
        analyzer = TenderAnalyzer()
        # Predefined queries with titles
        queries = {
            "What are the functional requirements, also known as the scope of work, mentioned in the document?": "Scope of Work",
            "Extract clauses that specify Pre-Qualification Criteria or eligibility criteria.": "Prequalification Criteria",
            "List all supporting documents required for this tender.": "Supporting Documents",
            "List of all the dates mentioned in the tender document which should include Bid submission end date or due date of tender, Bid validity, Opening date, closing date, pre bid meeting date, EMD amount,tender fee, tender value": "Important Dates",
            "Extract the contact details of the officer from this document, including their name, email ID, and contact number.": "Contact Details"
        }

        # Process each text file in the directory
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        # Read the entire file content
                        with open(file_path, 'r', encoding='utf-8') as f:
                            combined_text = f.read()

                        # Process each query
                        for query, title in queries.items():
                            try:
                                # Process the query
                                result, _ = analyzer.process_query(query, combined_text, tabId)
                                
                                # Prepare response in the original format
                                response = {
                                    'title': title,
                                    'response': result,
                                    'tabId': tabId
                                }
                                
                                # Add to results and yield
                                results.append(response)
                                yield response
                                
                                # Clear CUDA cache
                                torch.cuda.empty_cache()
                            
                            except Exception as e:
                                error_response = {
                                    'title': title,
                                    'error': str(e),
                                    'tabId': tabId
                                }
                                results.append(error_response)
                                yield error_response

                        # Index results into OpenSearch
                        doc_id = tcno
                        json_response = {"results": results}
                        index_results(tcno, results)
                        update_elasticsearch(tcno)
                    except Exception as e:
                        error_response = {
                            'title': 'File Processing Error',
                            'error': f'Error processing file {file}: {str(e)}',
                            'tabId': tabId
                        }
                        yield error_response
    
    except Exception as e:
        yield {
            'title': 'Overall Processing Error',
            'error': str(e),
            'tabId': tabId
        }
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_established', {'data': 'Connected to server'})



@socketio.on('process_pdf')
def handle_process_pdf(tc_no, tabId):
    print(f'Received process_pdf event with tc_no: {tc_no} and tabId: {tabId}')
    try:
        results = []
        
        # Check if tc_no is valid
        if not tc_no:
            socketio.emit('pdf_processed_error', json.dumps({
                'error': 'Missing tender number', 
                'tabId': tabId
            }))
            return

        # Process PDF and stream results
        for response in process_pdf(tc_no, tabId, results):
            try:
                # Ensure response is JSON serializable
                response_json = json.dumps(response, ensure_ascii=False)
                print("Streaming response:", response_json)
                
                # Emit each response individually
                socketio.emit('response', response_json)
            except Exception as e:
                print(f'Emission error for response: {e}')
                socketio.emit('pdf_processed_error', json.dumps({
                    'error': f'Emission error: {str(e)}', 
                    'tabId': tabId
                }))

        # Emit completion status after processing all documents
        socketio.emit('pdf_processed_complete', json.dumps({
            'status': 'complete', 
            'tabId': tabId,
            'total_results': len(results)
        }))

    except Exception as e:
        print(f'Error during PDF processing: {e}')
        socketio.emit('pdf_processed_error', json.dumps({
            'error': str(e), 
            'tabId': tabId
        }))

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')




class TenderAnalyzerQA:
    def __init__(self):
        # Update LLM configuration 
        try:
            self.llm = ChatOpenAI(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                openai_api_base="http://localhost:8000/v1",
                openai_api_key="FAKE",
                max_tokens=512,
                temperature=0.1
            )
            self.chain = load_qa_chain(self.llm, chain_type='stuff')
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            self.llm = None
            self.chain = None


    
    def googleOcr(self,imgfile):
        try:
            creds = service_account.Credentials.from_service_account_file(
                r"/data/imageExtraction/GoogleAPICred/projectoct-436907-a6e51afb9d49.json",
                scopes=['https://www.googleapis.com/auth/drive']
            )
            service = build('drive', 'v3', credentials=creds)
            mime = 'application/vnd.google-apps.document'
            res = service.files().create(
                body={
                    'name': imgfile,
                    'mimeType': mime
                },
                media_body=MediaFileUpload(imgfile, mimetype=mime, resumable=True)
            ).execute()
            
            text_output = io.BytesIO()
            downloader = MediaIoBaseDownload(
                text_output,
                service.files().export_media(fileId=res['id'], mimeType="text/plain")
            )
            done = False
            while not done:
                status, done = downloader.next_chunk()
            text_output.seek(0)
            extracted_text = text_output.read().decode('utf-8')
            service.files().delete(fileId=res['id']).execute()
            
            return extracted_text

        except Exception as e:
            logger.info(e)
            return None
    
    def process_documentQA(self, file_path: str) -> List[str]:
        """Process document and split into chunks"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Handle different file types based on extension
            file_type_handlers = {
                '.txt': self._read_txt,
                '.csv': self._read_csv,
                '.pdf': self._read_pdf,
                '.doc': self._read_docx,
                '.docx': self._read_docx,
                '.xls': self._read_excel,
                '.xlsx': self._read_excel,
                '.jpg': self._read_image,
                '.jpeg': self._read_image,
                '.png': self._read_image
            }
            
            handler = file_type_handlers.get(file_extension)
            if not handler:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            text = handler(file_path)
            sentences = self._split_into_sentences(text)
            chunks = self._create_chunks(sentences)
            return self._chunk_by_tokens(chunks)
        
        except Exception as e:
            print(f"Error processing document: {e}")
            return []

    def _read_txt(self, file_path: str) -> str:
        """Read a plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _read_csv(self, file_path: str) -> str:
        """Read a CSV file and return the text"""
        df = pd.read_csv(file_path)
        return df.to_string(index=False)

    def _read_pdf(self, file_path: str) -> str:
       """Read a PDF file using PyPDF2"""
       reader = PdfReader(file_path)
       text = ""
       for page in reader.pages:
          text += page.extract_text()
       return text

    def _read_docx(self, file_path: str) -> str:
        """Read a DOCX file"""
        doc = DocxDocument(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + '\n'
        return text

    def _read_excel(self, file_path: str) -> str:
        """Read an Excel file using pandas"""
        df = pd.read_excel(file_path)
        return df.to_string(index=False)
    
    def _read_image(self, file_path: str) -> str:
      """
      Process the uploaded image file to extract text using OCR.
      """
      extracted_text = self.googleOcr(file_path)
      return extracted_text if extracted_text else "No text extracted from image."


    def _split_into_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sentences with metadata"""
        # Use regex to handle various sentence endings
        sentences = [{'sentence': s.strip(), 'index': i} 
                    for i, s in enumerate(re.split(r'(?<=[.!?])\s+', text)) if s.strip()]
        return self._combine_sentences(sentences)

    def _combine_sentences(self, sentences: List[Dict[str, Any]], buffer_size: int = 1) -> List[Dict[str, Any]]:
        """Combine sentences with context"""
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
        """Create document chunks based on semantic similarity"""
        try:
            embeddings = [get_embeddings_via_api(s['combined_sentence']) for s in sentences]
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
        except Exception as e:
            print(f"Error in embeddings: {e}")
            # Fallback to original chunking if API fails
            return [' '.join([s['sentence'] for s in sentences])]

    def _chunk_by_tokens(self, texts: List[str], max_tokens: int = 3500) -> List[str]:
        """Split texts into smaller chunks based on character count"""
        max_chars = max_tokens * 2
        chunks = []
        for text in texts:
            text_chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
            chunks.extend(text_chunks)
        return chunks

    def process_queryQA(self, query: str, text: str) -> str:
        """Process a single query against the text"""
        if not self.chain:
            return "LLM not initialized"
        
        try:
            response = self.chain.run(input_documents=[Document(page_content=text)], question=query)
            return response.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def analyze_tenderQA(self, file_path: str, query: str) -> str:
        """Main analysis function for a custom query"""
        chunks = self.process_documentQA(file_path)
        if not chunks:
            return "Could not process the document"
        
        combined_text = " ".join(chunks)
        response = self.process_queryQA(query, combined_text)
        return response
    def generate_embeddings(self, file_path: str):
        """
        Generate and store embeddings for the uploaded document
        This method uses the existing document processing logic
        """
        try:
            # Process the document into chunks
            chunks = self.process_documentQA(file_path)
            
            # Store document details and embeddings globally
            uploaded_documents['file_path'] = file_path
            uploaded_documents['file_name'] = os.path.basename(file_path)
            
            # Store chunks for later use
            embeddings['chunks'] = chunks
            
            return chunks
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []
    # Initialize the TenderAnalyzer
tender_analyzer = TenderAnalyzerQA()


@app.route('/upload', methods=['POST'])
def upload_file():
    if not request.files:
        return jsonify({"error": "No files uploaded"}), 400
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    
    try:
        # Save the uploaded file
        file.save(file_path)

        # Generate embeddings or process the file
        tender_analyzer.generate_embeddings(file_path)

        return jsonify({
            "message": f"Document '{file.filename}' uploaded and embeddings generated successfully."
        }), 200
    except Exception as e:
        app.logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to upload the file", "details": str(e)}), 500



@app.route('/query', methods=['POST'])
def query_file():
    """Handle querying on the uploaded document."""
    if 'file_path' not in uploaded_documents:
        return jsonify({"error": "No document uploaded. Please upload a document first."}), 400

    query = request.form.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Retrieve precomputed embeddings and process the query
        file_name = uploaded_documents['file_name']
        chunks = embeddings.get('chunks', [])
        if not chunks:
            return jsonify({"error": f"Embeddings for document '{file_name}' are not available."}), 500

        combined_text = " ".join(chunks)
        response = tender_analyzer.process_queryQA(query, combined_text)
        return jsonify({"document": file_name, "response": response}), 200
    except Exception as e:
        return jsonify({"error": "Failed to process the query", "details": str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5003)
