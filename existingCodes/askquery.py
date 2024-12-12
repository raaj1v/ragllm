# import os
# import subprocess
# import pickle
# import base64
# from typing import List, Tuple
# from flask import Flask, request, jsonify
# import requests
# import pandas as pd
# from docx import Document as DocxDocument
# from tabulate import tabulate
# from PyPDF2 import PdfReader

# from langchain.chains.question_answering import load_qa_chain
# from langchain_openai import ChatOpenAI
# from langchain.callbacks import get_openai_callback
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.schema import Document as LangchainDocument
# from langchain.embeddings.base import Embeddings
# from langchain_community.document_loaders import TextLoader, PyPDFLoader

# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)

# # Suppress warnings
# import warnings
# warnings.filterwarnings("ignore")

# # Set environment variables (adjust as needed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

# # Initialize LLM
# llm = ChatOpenAI(
#     model_name="meta-llama/Meta-Llama-3-8B-Instruct",
#     openai_api_base="http://localhost:8000/v1",
#     openai_api_key="FAKE",  # Replace with your actual API key
#     max_tokens=512,
#     temperature=0.1
# )

# # Define Embedding functions and classes

# def get_embedding(text: str) -> List[float]:
#     """
#     Fetch embedding for the given text from the embedding API.
#     """
#     try:
#         response = requests.post(
#             "http://0.0.0.0:5002/embeddings",
#             json={"model": "BAAI/bge-small-en-v1.5", "input": [text]}
#         )
#         response.raise_for_status()
#         data = response.json()
#         return data['data'][0]['embedding']
#     except Exception as e:
#         logger.error(f"Failed to get embedding: {e}")
#         raise e

# class CustomEmbeddings(Embeddings):
#     """
#     Custom Embeddings class that utilizes an external API to generate embeddings.
#     """
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return [get_embedding(text) for text in texts]

#     def embed_query(self, text: str) -> List[float]:
#         return get_embedding(text)

# # Document Processing Functions

# def process_document(file_path: str) -> Tuple[str, str]:
#     """
#     Process .doc and .docx files to extract text and tables.
#     """
#     def convert_doc_to_txt(file_path, output_file):
#         # Convert .doc to .txt using LibreOffice
#         try:
#             subprocess.run(
#                 ['libreoffice', '--headless', '--convert-to', 'txt:Text', '--outdir', os.path.dirname(output_file), file_path],
#                 check=True
#             )
#             os.rename(file_path.replace('.doc', '.txt'), output_file)
#         except subprocess.CalledProcessError as e:
#             logger.error(f"LibreOffice conversion failed: {e}")
#             raise e

#     def read_docx(file_path1: str) -> Tuple[str, str]:
#         # Read text and tables from .docx file
#         doc = DocxDocument(file_path1)
#         text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])

#         # Extract tables
#         tables_data = []
#         for table in doc.tables:
#             table_data = []
#             for row in table.rows:
#                 row_data = [cell.text.strip() for cell in row.cells]
#                 table_data.append(row_data)
#             tables_data.append(table_data)
        
#         tables_text = "\n\n".join([tabulate(table, tablefmt="grid") for table in tables_data])
#         return text, tables_text

#     def read_txt(file_path1: str) -> str:
#         # Read text from .txt file
#         with open(file_path1, 'r', encoding='utf-8') as file:
#             return file.read()

#     output_file = file_path.replace('.doc', '.txt')

#     if file_path.endswith('.docx'):
#         text, tables_text = read_docx(file_path)
#         return text, tables_text
#     elif file_path.endswith('.doc'):
#         convert_doc_to_txt(file_path, output_file)
#         text = read_txt(output_file)
#         os.remove(output_file)
#         return text, ""
#     else:
#         raise ValueError("Unsupported file format. Only .doc and .docx are supported.")

# def process_text(texts: List[LangchainDocument]) -> FAISS:
#     """
#     Process and split text into chunks, then create a FAISS index.
#     """
#     combined_text = "\n".join([doc.page_content for doc in texts])
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=2048,
#         chunk_overlap=32,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(combined_text)

#     embeddings = CustomEmbeddings()
#     knowledge_base = FAISS.from_texts(chunks, embedding=embeddings)
#     return knowledge_base

# def save_embedding(tcno: str, knowledge_base: FAISS):
#     """
#     Save the FAISS index to disk using pickle.
#     """
#     embeddings_dir = 'embeddings/'
#     os.makedirs(embeddings_dir, exist_ok=True)  # Create directory if it doesn't exist
#     save_path = os.path.join(embeddings_dir, f"{tcno}.pkl")
#     try:
#         with open(save_path, 'wb') as f:
#             pickle.dump(knowledge_base, f)
#         logger.info(f"Embeddings saved for TCNO {tcno} at `{save_path}`")
#     except Exception as e:
#         logger.error(f"Failed to save embeddings: {e}")
#         raise e

# def load_embedding(tcno: str) -> FAISS:
#     """
#     Load the FAISS index from disk if it exists.
#     """
#     load_path = os.path.join('embeddings/', f"{tcno}.pkl")
#     if os.path.exists(load_path):
#         try:
#             with open(load_path, 'rb') as f:
#                 knowledge_base = pickle.load(f)
#             logger.info(f"Loaded embeddings for TCNO {tcno} from `{load_path}`")
#             return knowledge_base
#         except Exception as e:
#             logger.error(f"Failed to load embeddings: {e}")
#             return None
#     else:
#         logger.warning(f"No embeddings found for TCNO {tcno}")
#         return None

# def load_text_files(uploaded_files: List, tcno: str) -> List[LangchainDocument]:
#     """
#     Load and process uploaded text, PDF, DOC, DOCX, CSV, and Excel files.
#     """
#     all_docs = []
#     temp_dir = "temp_files"
#     os.makedirs(temp_dir, exist_ok=True)

#     for uploaded_file in uploaded_files:
#         try:
#             file_extension = os.path.splitext(uploaded_file.filename)[1].lower()
#             temp_path = os.path.join(temp_dir, uploaded_file.filename)
            
#             with open(temp_path, "wb") as f:
#                 f.write(uploaded_file.read())
            
#             if file_extension == ".txt":
#                 loader = TextLoader(temp_path)
#                 docs = loader.load()
#                 all_docs.extend(docs)
#                 logger.info(f"Loaded text file: `{uploaded_file.filename}`")
            
#             elif file_extension == ".pdf":
#                 loader = PyPDFLoader(temp_path)
#                 docs = loader.load()
#                 all_docs.extend(docs)
#                 logger.info(f"Loaded PDF file: `{uploaded_file.filename}`")
            
#             elif file_extension in [".doc", ".docx"]:
#                 text, tables_text = process_document(temp_path)
#                 combined_text = text
#                 if tables_text:
#                     combined_text += "\n" + tables_text
#                 langchain_doc = LangchainDocument(page_content=combined_text)
#                 all_docs.append(langchain_doc)
#                 logger.info(f"Loaded document file: `{uploaded_file.filename}`")
            
#             elif file_extension == ".csv":
#                 df = pd.read_csv(temp_path)
#                 langchain_doc = LangchainDocument(page_content=df.to_string(index=False))
#                 all_docs.append(langchain_doc)
#                 logger.info(f"Loaded CSV file: `{uploaded_file.filename}`")
            
#             elif file_extension in [".xls", ".xlsx"]:
#                 df = pd.read_excel(temp_path)
#                 langchain_doc = LangchainDocument(page_content=df.to_string(index=False))
#                 all_docs.append(langchain_doc)
#                 logger.info(f"Loaded Excel file: `{uploaded_file.filename}`")
            
#             else:
#                 logger.warning(f"Unsupported file type: `{uploaded_file.filename}`")
#         except Exception as e:
#             logger.error(f"Error loading `{uploaded_file.filename}`: {e}")
#     return all_docs

# def process_query(query: str, knowledge_base: FAISS) -> str:
#     """
#     Process the user's query against the knowledge base and return the response.
#     """
#     try:
#         docs = knowledge_base.similarity_search(query, k=10)
#         chain = load_qa_chain(llm, chain_type='stuff')
        
#         instruction = (
#             "Extract all relevant fields from the provided document and "
#             "answer the input query based only on the dataset. "
#             "Do not include irrelevant information."
#         )

#         full_input = f"{instruction}\n\nQuery: {query}"

#         with get_openai_callback() as cost:
#             response = chain.run(input_documents=docs, question=full_input)
#             logger.info(f"OpenAI Cost: {cost}")

#         return response
#     except Exception as e:
#         logger.error(f"Error processing query: {e}")
#         return f"Error processing query: {e}"

# # Flask Endpoints

# @app.route('/upload', methods=['POST'])
# def upload_documents():
#     """
#     Endpoint to upload documents and create a knowledge base.
#     """
#     logger.info("Received upload request.")
#     if 'tcno' not in request.form or 'files' not in request.files:
#         logger.error("Missing 'tcno' or 'files' in the request.")
#         return jsonify({"error": "TCNO and files are required."}), 400

#     tcno = request.form['tcno']
#     uploaded_files = request.files.getlist('files')

#     if not uploaded_files:
#         logger.error("No files uploaded.")
#         return jsonify({"error": "No files uploaded."}), 400

#     logger.info(f"TCNO: {tcno}")
#     logger.info(f"Number of files received: {len(uploaded_files)}")

#     all_docs = load_text_files(uploaded_files, tcno)

    # if all_docs:
    #     try:
    #         # Create knowledge base
    #         knowledge_base = process_text(all_docs)
    #         # Save knowledge base
    #         save_embedding(tcno, knowledge_base)
    #         logger.info("Documents processed and embeddings are ready.")
    #         return jsonify({"message": "Documents processed and knowledge base created successfully."}), 200
    #     except Exception as e:
    #         logger.error(f"Failed to create knowledge base: {e}")
    #         return jsonify({"error": f"Failed to create knowledge base: {e}"}), 500
    # else:
    #     logger.error("No documents were loaded.")
    #     return jsonify({"error": "No documents were loaded."}), 400

# @app.route('/query', methods=['POST'])
# def query_endpoint():
#     """
#     Endpoint to process queries against the knowledge base.
#     """
#     logger.info("Received query request.")
#     data = request.get_json()
#     if not data:
#         logger.error("No JSON data provided.")
#         return jsonify({"error": "No JSON data provided."}), 400

#     if 'tcno' not in data or 'query' not in data:
#         logger.error("Missing 'tcno' or 'query' in the request.")
#         return jsonify({"error": "TCNO and query are required."}), 400

#     tcno = data['tcno']
#     query_text = data['query']

#     logger.info(f"TCNO: {tcno}")
#     logger.info(f"Query: {query_text}")

#     knowledge_base = load_embedding(tcno)
#     if not knowledge_base:
#         logger.error("Knowledge base not found.")
#         return jsonify({"error": "Knowledge base not found."}), 404

#     response = process_query(query_text, knowledge_base)
#     return jsonify({"response": response}), 200

# # Utility Functions (Optional Enhancements)

# def cleanup_temp_files():
#     """
#     Clean up temporary files after processing.
#     """
#     temp_dir = "temp_files"
#     if os.path.exists(temp_dir):
#         for filename in os.listdir(temp_dir):
#             file_path = os.path.join(temp_dir, filename)
#             try:
#                 if os.path.isfile(file_path):
#                     os.unlink(file_path)
#                     logger.info(f"Deleted temporary file: `{file_path}`")
#             except Exception as e:
#                 logger.warning(f"Failed to delete `{file_path}`: {e}")

# # Run the Flask app
# if __name__ == '__main__':
#     os.makedirs('temp_files', exist_ok=True)  # Create temp directory if it doesn't exist
#     os.makedirs('embeddings', exist_ok=True)  # Create embeddings directory if it doesn't exist
#     app.run(host='0.0.0.0', port=5010)

import sys, ast
sys.path.append(r'/data/imageExtraction/')
sys.path.append(r'/data/imageExtraction/imgEnv/')
sys.path.append(r'/data/QAAPI/')
sys.path.append(r'/data/QAAPI/qavenv/')
import os
import subprocess
import pickle
import base64
from typing import List, Tuple
from flask import Flask, request, jsonify
import requests
import pandas as pd
# from docx import Document as DocxDocument
from tabulate import tabulate
from PyPDF2 import PdfReader

from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document as LangchainDocument
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set environment variables (adjust as needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
from tabulate import tabulate
from PyPDF2 import PdfReader
from google.oauth2 import service_account
from googleapiclient.discovery import build
from apiclient.http import MediaFileUpload, MediaIoBaseDownload
from deep_translator import GoogleTranslator
from langdetect import detect
import io
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set environment variables (adjust as needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Initialize LLM
llm = ChatOpenAI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="FAKE",  # Replace with your actual API key
    max_tokens=512,
    temperature=0.1
)

# Define Embedding functions and classes
def get_embedding(text: str) -> List[float]:
    try:
        response = requests.post(
            "http://0.0.0.0:5002/embeddings",
            json={"model": "BAAI/bge-small-en-v1.5", "input": [text]}
        )
        response.raise_for_status()
        data = response.json()
        return data['data'][0]['embedding']
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        raise e

class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        max_workers = min(80, len(texts))  # Use up to 80 workers or fewer if fewer texts\
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            embeddings = list(executor.map(get_embedding, texts))
        return embeddings    
        # return [get_embedding(text) for text in texts]
     
    def embed_query(self, text: str) -> List[float]:
        return get_embedding(text)

def process_text(texts: List[LangchainDocument]) -> FAISS:
    """
    Process and split text into chunks, then create a FAISS index.
    """
    combined_text = "\n".join([doc.page_content for doc in texts])
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2048,
        chunk_overlap=32,
        length_function=len
    )
    chunks = text_splitter.split_text(combined_text)

    embeddings = CustomEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embedding=embeddings)
    return knowledge_base


def save_embedding(tcno: str, knowledge_base: FAISS):
    """
    Save the FAISS index to disk using pickle.
    """
    embeddings_dir = 'embeddings/'
    os.makedirs(embeddings_dir, exist_ok=True)  # Create directory if it doesn't exist
    save_path = os.path.join(embeddings_dir, f"{tcno}.pkl")
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(knowledge_base, f)
        logger.info(f"Embeddings saved for TCNO {tcno} at `{save_path}`")
    except Exception as e:
        logger.error(f"Failed to save embeddings: {e}")
        raise e

def process_document(file_path: str) -> Tuple[str, str]:
    def convert_doc_to_txt(file_path, output_file):
        try:
            subprocess.run(
                ['libreoffice', '--headless', '--convert-to', 'txt:Text', '--outdir', os.path.dirname(output_file), file_path],
                check=True
            )
            os.rename(file_path.replace('.doc', '.txt'), output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"LibreOffice conversion failed: {e}")
            raise e

    def read_docx(file_path1: str) -> Tuple[str, str]:
        doc = DocxDocument(file_path1)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        tables_data = []
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)
            tables_data.append(table_data)
        
        tables_text = "\n\n".join([tabulate(table, tablefmt="grid") for table in tables_data])
        return text, tables_text

    def read_txt(file_path1: str) -> str:
        with open(file_path1, 'r', encoding='utf-8') as file:
            return file.read()

    output_file = file_path.replace('.doc', '.txt')

    if file_path.endswith('.docx'):
        text, tables_text = read_docx(file_path)
        return text, tables_text
    elif file_path.endswith('.doc'):
        convert_doc_to_txt(file_path, output_file)
        text = read_txt(output_file)
        os.remove(output_file)
        return text, ""
    else:
        raise ValueError("Unsupported file format. Only .doc and .docx are supported.")

def googleOcr(imgfile):
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

def process_image(file_path: str) -> str:
    """
    Process the uploaded image file to extract text using OCR.
    """
    extracted_text = googleOcr(file_path)
    return extracted_text if extracted_text else "No text extracted from image."


def load_embedding(tcno: str) -> FAISS:
    """
    Load the FAISS index from disk if it exists.
    """
    load_path = os.path.join('embeddings/', f"{tcno}.pkl")
    if os.path.exists(load_path):
        try:
            with open(load_path, 'rb') as f:
                knowledge_base = pickle.load(f)
            logger.info(f"Loaded embeddings for TCNO {tcno} from `{load_path}`")
            return knowledge_base
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return None
    else:
        logger.warning(f"No embeddings found for TCNO {tcno}")
        return None


def load_text_files(uploaded_files: List, tcno: str) -> List[LangchainDocument]:
    all_docs = []
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        try:
            file_extension = os.path.splitext(uploaded_file.filename)[1].lower()
            temp_path = os.path.join(temp_dir, uploaded_file.filename)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            if file_extension == ".txt":
                loader = TextLoader(temp_path)
                docs = loader.load()
                all_docs.extend(docs)
                logger.info(f"Loaded text file: `{uploaded_file.filename}`")
            
            elif file_extension == ".pdf":
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                all_docs.extend(docs)
                logger.info(f"Loaded PDF file: `{uploaded_file.filename}`")
            
            elif file_extension in [".doc", ".docx"]:
                text, tables_text = process_document(temp_path)
                combined_text = text
                if tables_text:
                    combined_text += "\n" + tables_text
                langchain_doc = LangchainDocument(page_content=combined_text)
                all_docs.append(langchain_doc)
                logger.info(f"Loaded document file: `{uploaded_file.filename}`")
            
            elif file_extension == ".csv":
                df = pd.read_csv(temp_path)
                langchain_doc = LangchainDocument(page_content=df.to_string(index=False))
                all_docs.append(langchain_doc)
                logger.info(f"Loaded CSV file: `{uploaded_file.filename}`")
            
            elif file_extension in [".xls", ".xlsx"]:
                df = pd.read_excel(temp_path)
                langchain_doc = LangchainDocument(page_content=df.to_string(index=False))
                all_docs.append(langchain_doc)
                logger.info(f"Loaded Excel file: `{uploaded_file.filename}`")

            elif file_extension in [".jpg", ".jpeg", ".png"]:
                extracted_text = process_image(temp_path)
                langchain_doc = LangchainDocument(page_content=extracted_text)
                all_docs.append(langchain_doc)
                logger.info(f"Loaded image file: `{uploaded_file.filename}`")
            
            else:
                logger.warning(f"Unsupported file type: `{uploaded_file.filename}`")
        except Exception as e:
            logger.error(f"Error loading `{uploaded_file.filename}`: {e}")
    return all_docs

# def process_query(query: str, knowledge_base: FAISS) -> str:
#     """
#     Process the user's query against the knowledge base and return the response.
#     """
#     try:
#         docs = knowledge_base.similarity_search(query, k=10)
#         chain = load_qa_chain(llm, chain_type='stuff')
        
#         instruction = (
#             "Extract all relevant fields from the provided document and "
#             "answer the input query based only on the dataset. "
#             "Do not include irrelevant information."
#         )

#         full_input = f"{instruction}\n\nQuery: {query}"

#         with get_openai_callback() as cost:
#             response = chain.run(input_documents=docs, question=full_input)
#             logger.info(f"OpenAI Cost: {cost}")

#         return response
#     except Exception as e:
#         logger.error(f"Error processing query: {e}")
#         return f"Error processing query: {e}"

def process_query(query: str, knowledge_base: FAISS) -> str:
    """
    Process the user's query against the knowledge base and return the response using parallel processing.
    """
    try:
        # Get top 10 similar documents using parallel processing
        def search_similar_docs(doc):
            return knowledge_base.similarity_search(doc, k=10)

        # Split the query into multiple chunks (if needed)
        docs = knowledge_base.similarity_search(query, k=10)

        # Use ThreadPoolExecutor for parallel processing of query results
        max_workers = min(80, len(docs))  # Use up to 80 workers or fewer depending on the number of docs
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            search_results = list(executor.map(search_similar_docs, [query]*len(docs)))

        # Flatten the list of results
        flattened_results = [doc for result in search_results for doc in result]

        # Load the QA chain
        chain = load_qa_chain(llm, chain_type='stuff')

        # Instruction for the model
        instruction = (
            "Extract all relevant fields from the provided document and "
            "answer the input query based only on the dataset. "
            "Do not include irrelevant information."
        )

        full_input = f"{instruction}\n\nQuery: {query}"

        # Process the query with the model using the parallel results
        with get_openai_callback() as cost:
            response = chain.run(input_documents=flattened_results, question=full_input)
            logger.info(f"OpenAI Cost: {cost}")

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"Error processing query: {e}"


# Flask Endpoints
@app.route('/upload', methods=['POST'])
def upload_documents():
    logger.info("Received upload request.")
    if 'tcno' not in request.form or 'files' not in request.files:
        logger.error("Missing 'tcno' or 'files' in the request.")
        return jsonify({"error": "TCNO and files are required."}), 400

    tcno = request.form['tcno']
    uploaded_files = request.files.getlist('files')

    if not uploaded_files:
        logger.error("No files uploaded.")
        return jsonify({"error": "No files uploaded."}), 400

    logger.info(f"TCNO: {tcno}")
    logger.info(f"Number of files received: {len(uploaded_files)}")

    all_docs = load_text_files(uploaded_files, tcno)

    if all_docs:
        try:
            knowledge_base = process_text(all_docs)
            save_embedding(tcno, knowledge_base)
            logger.info("Documents processed and embeddings are ready.")
            return jsonify({"message": "Documents processed and knowledge base created successfully."}), 200
        except Exception as e:
            logger.error(f"Failed to create knowledge base: {e}")
            return jsonify({"error": f"Failed to create knowledge base: {e}"}), 500
    else:
        logger.error("No documents were loaded.")
        return jsonify({"error": "No documents were loaded."}), 400
    
@app.route('/query', methods=['POST'])
def query_endpoint():
    """
    Endpoint to process queries against the knowledge base.
    """
    logger.info("Received query request.")
    data = request.get_json()
    if not data:
        logger.error("No JSON data provided.")
        return jsonify({"error": "No JSON data provided."}), 400

    if 'tcno' not in data or 'query' not in data:
        logger.error("Missing 'tcno' or 'query' in the request.")
        return jsonify({"error": "TCNO and query are required."}), 400

    tcno = data['tcno']
    query_text = data['query']

    logger.info(f"TCNO: {tcno}")
    logger.info(f"Query: {query_text}")

    knowledge_base = load_embedding(tcno)
    if not knowledge_base:
        logger.error("Knowledge base not found.")
        return jsonify({"error": "Knowledge base not found."}), 404

    response = process_query(query_text, knowledge_base)
    return jsonify({"response": response}), 200

# Utility Functions (Optional Enhancements)

def cleanup_temp_files():
    """
    Clean up temporary files after processing.
    """
    temp_dir = "temp_files"
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logger.info(f"Deleted temporary file: `{file_path}`")
            except Exception as e:
                logger.warning(f"Failed to delete `{file_path}`: {e}")

# Run the Flask app
if __name__ == '__main__':
    os.makedirs('temp_files', exist_ok=True)  # Create temp directory if it doesn't exist
    os.makedirs('embeddings', exist_ok=True)  # Create embeddings directory if it doesn't exist
    app.run(host='0.0.0.0', port=5010)

