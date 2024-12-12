


# import os
# import warnings
# import json
# import pickle  # For saving and loading embeddings
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain_openai import ChatOpenAI
# from langchain.callbacks import get_openai_callback
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain_community.document_loaders import TextLoader
# import requests
# from typing import List
# from langchain.vectorstores import FAISS
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.base import Embeddings

# import torch
# import requests
# from typing import List

# # Set environment variables
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"
# warnings.filterwarnings("ignore")

# # Define Flask app
# app = Flask(__name__)
# CORS(app)

# # # Query titles
# # query_titles = {
# #     "bid_date": "Extract only the bid end date and time from the document. (provide only bid date without any extra details)",
# #     "ministry_state_name": "Extract only the ministry/state name from the document. (provide only ministry or state name without any extra details)",
# #     "item_category": "Extract the list of all 'Item Category' sections from the tender document. Provide only the 'Item Category' text without any additional details or context.",
# #     "bid_number": "Please extract the Bid Number from the tender document., with no extra details.",
# #     "contract_period":"Extract only the contract period from the document. (provide only contract period without any extra details)",
# #     "Beneficiary_details":"details of Beneficiary mentioned in this tender?",
# #     "bid_type": "Extract only the type of bid from the document. (provide only type of bid without any extra details)",
# #     "Emd_detail":"EMD detail with bank name mentioned in this tender?",
# #     "Estimated_bid_value":"extract the estimated bid value from the tender document.(provide only estimated bid value without any extra details)",
# #     "EMD EXEMPTION":"Extract the full details of emd exemption from the tender document.(provide only emd exemption without any extra details)"
# # }

# # Query titles
# query_titles = {
#     "bid_date": "Extract only the bid end date and time from the document. Provide only the bid date without any extra details.",
#     "ministry_state_name": "Extract only the ministry or state name from the document. Provide only the ministry or state name without any extra details.",
#     "Department_Name": "Extract only the Department Name from the document. Provide only the Department Name without any extra details.",
#     "Organisation_Name":"Extract only the Organisation Name from the document. Provide only the Organisation Name without any extra details.",
#     "item_category": "Extract the list of all 'Item Category' sections from the tender document. Provide only the 'Item Category' text without any additional details.",
#     "bid_number": "Extract the Bid Number from the tender document. Provide only the Bid Number without any extra details.",
#     "contract_period": "Extract only the contract period from the document. Provide only the contract period without any extra details.",
#     "Beneficiary_details": "Extract the details of the Beneficiary mentioned in the tender.",
#     "bid_type": "Extract only the type of bid from the document. Provide only the type of bid without any extra details.",
#     "Emd_detail": "Extract the EMD detail with the bank name mentioned in the tender.",
#     "Estimated_bid_value": "Extract the estimated bid value from the tender document. Provide only the estimated bid value without any extra details.",
#     "EMD_EXEMPTION": "Extract the full details of the EMD exemption from the tender document. Provide only the EMD exemption details without any extra information."
# }

# llm = ChatOpenAI(
#     model_name="meta-llama/Meta-Llama-3-8B-Instruct",
#     openai_api_base="http://localhost:8000/v1",
#     openai_api_key="FAKE",
#     max_tokens=256,
#     temperature=0.1
# )

# def get_embedding(text: str) -> List[float]:
#     response = requests.post("http://0.0.0.0:5002/embeddings",
#         json={"model": "BAAI/bge-small-en-v1.5", "input": [text]})
#     if response.status_code == 200:
#         data = response.json()
#         return data['data'][0]['embedding']
#     else:
#         raise Exception(f"API request failed with status code {response.status_code}")

# class CustomEmbeddings(Embeddings):
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return [get_embedding(text) for text in texts]

#     def embed_query(self, text: str) -> List[float]:
#         return get_embedding(text)

# def process_text(text):
#     text = "\n".join([doc.page_content for doc in text])
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=2048,
#         chunk_overlap=32,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
    
#     embeddings = CustomEmbeddings()
#     knowledgeBase = FAISS.from_texts(chunks, embedding=embeddings)
#     return knowledgeBase

# def save_embedding(tcno, knowledge_base):
#     embeddings_dir = '/data/tendergpt/embeddings/'
#     os.makedirs(embeddings_dir, exist_ok=True)  # Create directory if it doesn't exist
#     save_path = f"/data/tendergpt/embeddings/{tcno}.pkl"
#     with open(save_path, 'wb') as f:
#         pickle.dump(knowledge_base, f)
#     print(f"Embeddings saved for tcno {tcno} at {save_path}")

# def load_embedding(tcno):
#     load_path = f"/data/tendergpt/embeddings/{tcno}.pkl"
#     if os.path.exists(load_path):
#         with open(load_path, 'rb') as f:
#             knowledge_base = pickle.load(f)
#         print(f"Loaded embeddings for tcno {tcno} from {load_path}")
#         return knowledge_base
#     else:
#         return None

# def load_text_files_from_directory(folder_path):
#     all_docs = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(folder_path, filename)
#             loader = TextLoader(file_path)
#             all_docs.extend(loader.load())
#     return all_docs

# @app.route('/process_query', methods=['POST'])
# def process_query():
#     try:
#         tcno = request.json.get('tcno')
#         query_key = request.json.get('query_key')

#         if not tcno or not query_key:
#             return jsonify({'error': 'Missing tcno or query key'}), 400

#         # Load or generate embeddings
#         knowledge_base = load_embedding(tcno)
#         if not knowledge_base:
#             folder_path = f"/data/tendergpt/livetender_txt/{tcno}"
#             all_docs = load_text_files_from_directory(folder_path)
#             knowledge_base = process_text(all_docs)
#             save_embedding(tcno, knowledge_base)

#         query = query_titles.get(query_key, "")
#         print("query::",query)
#         if not query:
#             return jsonify({'error': 'Invalid query key'}), 400

#         result = process_query_internal(query, knowledge_base)
#         return jsonify({'response': result})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# def process_query_internal(query, knowledge_base):
#     docs = knowledge_base.similarity_search(query, k=10)
#     chain = load_qa_chain(llm, chain_type='stuff')
    
#     instruction = (
#             "Extract all relevant fields from the provided document and "
#             "try to answer the input query solely on the dataset. "
#             "Do not include any irrelevant information or extra details."
#         )

#         # Combine the instruction with the query
#     full_input = f"{instruction}\n\nQuery: {query}"
# #     full_input = f"Extract relevant fields based on the query.\n\nQuery: {query}"
#     response = chain.run(input_documents=docs, question=full_input)
#     return response

# if __name__ == '__main__':
#     app.run(debug=False, host='0.0.0.0', port=5000)


import os
import warnings
import json
import pickle  # For saving and loading embeddings
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
import requests
from typing import List
from langchain.embeddings.base import Embeddings
import torch
import requests
from typing import List

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
warnings.filterwarnings("ignore")

# Define Flask app
app = Flask(__name__)
CORS(app)

# # Query titles
# query_titles = {
#     "bid_date": "Extract only the bid end date and time as listed in the document. Ensure the format follows the date and time provided, without any additional information.",
#     "ministry_state_name": "Extract only the exact ministry or state name as mentioned in the document. Do not include any extra details or variations.",
#     "Department_Name": "Extract the precise Department Name from the document. Provide the department's full name without any alterations or additional details.",
#     "Organisation_Name": "Extract only the exact Organisation Name from the document. Ensure there are no abbreviations or additional context included.",
#     "item_category": "List all instances of 'Item Category' mentioned in the document. Extract only the exact text under 'Item Category' without further context or details.",
#     "bid_number": "Extract the specific Bid Number from the document exactly as it appears. Do not include any surrounding text or extra details.",
#     "contract_period": "Extract only the contract period, including the start and end dates, as specified in the document. Ensure no additional information is included.",
#     "Beneficiary_details": "Extract the full details of the Beneficiary mentioned in the tender, including relevant information like name, designation, and address if available.",
#     "bid_type": "Extract only the specific type of bid from the document, such as 'Open' or 'Limited'. Provide no extra context or details.",
#     "Emd_detail": "Extract the EMD details, specifically mentioning the bank name and EMD amount as found in the document. Exclude any extra commentary or context.",
#     "Estimated_bid_value": "Extract the exact estimated bid value from the document. Provide only the numerical value without additional text or interpretation.",
#     "EMD_EXEMPTION": "Extract the complete details of the EMD exemption as specified in the document. Provide only the exemption details without unrelated information."
# }


# Query titles
query_titles = {
    "bid_date": "Extract only the bid end date and time from the document. Provide only the bid date without any extra details.",
    "ministry_state_name":"Extract only the ministry or state name from the document. Provide only the ministry or state name without any extra details.",
    "Department_Name": "Extract only the Department Name from the document. Provide only the Department Name without any extra details.",
    "Organisation_Name": "Extract only the Organisation Name from the document. Provide only the Organisation Name without any extra details.",
    "item_category": "Extract the list of all 'Item Category' sections from the tender document. Provide only the 'Item Category' text without any additional details.",
    "bid_number": "Extract the Bid Number from the tender document. Provide only the Bid Number without any extra details.",
    "contract_period": "Extract only the contract period from the document. Provide only the contract period without any extra details.",
    "Beneficiary_details": "Extract the details of the Beneficiary mentioned in the tender.",
    "bid_type": "Extract only the type of bid from the document. Provide only the type of bid without any extra details.",
    "Emd_detail": "Extract the EMD detail with the bank name mentioned in the tender.",
    "Estimated_bid_value": "Extract the estimated Bid Value   from the tender document. Provide only the estimated bid value without any extra details.",
    "EMD_EXEMPTION": "Extract the full details of the EMD exemption from the tender document. Provide only the EMD exemption details without any extra information."
}

llm = ChatOpenAI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="FAKE",
    max_tokens=256,
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

def process_text(text):
    text = "\n".join([doc.page_content for doc in text])
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2048,
        chunk_overlap=1024,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = CustomEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embedding=embeddings)
    return knowledgeBase

def save_embedding(tcno, knowledge_base):
    embeddings_dir = '/data/tendergpt/embeddings/'
    os.makedirs(embeddings_dir, exist_ok=True)  # Create directory if it doesn't exist
    save_path = f"/data/tendergpt/embeddings/{tcno}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(knowledge_base, f)
    print(f"Embeddings saved for tcno {tcno} at {save_path}")

def load_embedding(tcno):
    load_path = f"/data/tendergpt/embeddings/{tcno}.pkl"
    if os.path.exists(load_path):
        with open(load_path, 'rb') as f:
            knowledge_base = pickle.load(f)
        print(f"Loaded embeddings for tcno {tcno} from {load_path}")
        return knowledge_base
    else:
        return None

def load_text_files_from_directory(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path)
            all_docs.extend(loader.load())
    return all_docs

@app.route('/process_query', methods=['POST'])
def process_query():
    try:
        tcno = request.json.get('tcno')
        query_keys = request.json.get('query_keys', [])  # Expecting a list of query keys

        if not tcno or not query_keys:
            return jsonify({'error': 'Missing tcno or query keys'}), 400

        # Load or generate embeddings
        knowledge_base = load_embedding(tcno)
        if not knowledge_base:
            folder_path = f"/data/tendergpt/livetender_txt/{tcno}"
            all_docs = load_text_files_from_directory(folder_path)
            knowledge_base = process_text(all_docs)
            save_embedding(tcno, knowledge_base)

        responses = {}

        # Process each query key
        for query_key in query_keys:
            query = query_titles.get(query_key, "")
            if not query:
                responses[query_key] = 'Invalid query key'
            else:
                result = process_query_internal(query, knowledge_base)
                responses[query_key] = result

        return jsonify({'responses': responses})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_query_internal(query, knowledge_base):
    docs = knowledge_base.similarity_search(query)
    chain = load_qa_chain(llm, chain_type='stuff')
    
    # instruction = (
            # "Extract all relevant fields from the provided document and "
            # "try to answer the input query solely on the dataset. "
            # "Do not include any irrelevant information or extra details."
        # )
    instruction = (
            "Extract exact data from the provided document and "
            "answer the input query based only on the dataset. "
            "Do not include irrelevant or additional information."
        )
    # Combine the instruction with the query
    full_input = f"{instruction}\n\nQuery: {query}"
    response = chain.run(input_documents=docs, question=full_input)
    return response

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
