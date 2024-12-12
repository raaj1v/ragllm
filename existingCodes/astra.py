

import os
import torch
import warnings
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
warnings.filterwarnings("ignore")

# Global initialization of models
embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
llm_model = ChatOpenAI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="FAKE",
    max_tokens=512,
    temperature=0.1
)


# query_tittles = {
#     "Extract the name of the firm/company from the document (provide only the name of the company with no extra context)": "Name of company",
#     "Extract the year of incorporation in India of the entity (provide only the year with no extra context)": "Year of Incorporation in India",
#     "Extract the  list of names of the partners/directors of the entity (provide only the names of the partners with no extra context)": "Names of the Partners",
#     "Extract the complete  name and address of the principal banker of the entity (provide only the name and address with no extra context)": "Name and Address of the Principal Banker",
#     # "Extract the address of the firm/company (provide only the address with no extra context)": "Address of company",
#     "Extract the head office of the entity (provide only the head office with no extra context)":"Head office",
#     "Extract the local office of the entity (provide only the local office with no extra context)": "Local office",
#     "Extract the contact person of the entity with their name and designation (provide only the name and designation with no extra context)": "Name and Designation",
#     "Extract the telephone number of the entity (provide only the telephone number with no extra context)": "Telephone number",
#     "Extract the e-mail ID of the entity (provide only the e-mail ID with no extra context)": "E-mail ID",
#     "Extract the mobile number of the entity (provide only the mobile number with no extra context)": "Mobile No",
#     "Extract the landline number of the entity (provide only the landline number with no extra context)": "Landline No.",
#     "Extract the address for communication (with pin code) of the entity (provide only the address with pin code and no extra context)": "Address for communication",
#     "What is the GST number of the entity (provide only the GST number with no extra context)": "GST Number",
#     "What is the PAN number of the entity (provide only the PAN number with no extra context)": "PAN Number",
#     "Extract the full bidder’s bank details/entity bank details, including bank name, branch, account number, and IFSC code (provide only the full details with no extra context)": "Full Bidder’s Bank Details"
# }


query_tittles = {
    "Extract the name of the firm/company from the document (provide only the name  with no extra context)": "Name of company",
    "Extract the year of incorporation in India of silver touch technologies limited (provide only the year with no extra context)": "Year of Incorporation in India",
    "Extract the list of all names of the partners/directors of silver touch technologies limited (provide only the names of the partners with no extra context)": "Names of the Partners",
    "Extract the complete name and address of the principal banker of silver touch technologies limited (provide only the name and address with no extra context)": "Name and Address of the Principal Banker",
    "Extract the  head office address of silver touch technologies limited (provide only the head office with no extra context)": "Head office",
    # "Extract the local office of silver touch technologies limited (provide only the local office with no extra context)": "Local office",
    "Extract the contact person of silver touch technologies limited with their name and designation (provide only the name and designation with no extra context)": "Name and Designation",
    "Extract the telephone number of silver touch technologies limited (provide only the telephone number with no extra context)": "Telephone number",
    "Extract the e-mail ID of silver touch technologies limited (provide only the e-mail address with no extra context)": "E-mail Address",
    "Extract the mobile number of silver touch technologies limited (provide only the mobile number with no extra context)": "Mobile No",
    "Extract the landline number of silver touch technologies limited (provide only the landline number with no extra context)": "Landline No.",
    "Extract the address for communication (with pin code) of silver touch technologies limited(provide only the address with pin code and no extra context)": "Address for communication",
    "What is the GST number of silver touch technologies limited (provide only the GST number with no extra context)": "GST Number",
    "What is the PAN number of silver touch technologies limited (provide only the PAN number with no extra context)": "PAN Number",
    "Extract the full bidder’s bank details/silver touch technologies limited's bank details, including bank name, branch, account number, and IFSC code (provide only the full details with no extra context)": "Full Bidder’s Bank Details"
}


import requests
from typing import List
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.base import Embeddings

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
        chunk_overlap=32,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = CustomEmbeddings()
    
    knowledgeBase = FAISS.from_texts(chunks, embedding=embeddings)
    return knowledgeBase

# def process_text(text):
#     text = "\n".join([doc.page_content for doc in text])
#     text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1024, chunk_overlap=32)
#     chunks = text_splitter.split_text(text)
#     knowledge_base = FAISS.from_texts(chunks, embedding_model)
#     torch.cuda.empty_cache()  # Clear CUDA cache after embedding processing
#     return knowledge_base

def load_text_files_from_directory(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path)
            all_docs.extend(loader.load())
    return all_docs

def process_query(query, knowledge_base):
    try:
        docs = knowledge_base.similarity_search(query)
        chain = load_qa_chain(llm_model, chain_type='stuff')
        with get_openai_callback() as cost:
            response = chain.run(input_documents=docs, question=query)
        torch.cuda.empty_cache()  # Clear CUDA cache after model inference
        return {'title': query_tittles[query], 'response': response}
    except Exception as e:
        torch.cuda.empty_cache()  # Ensure cache is cleared even on error
        return {'title': query_tittles[query], 'error': str(e)}

def save_tcno_to_excel(tcno, results):
    data = []

    for res in results:
        data.append({
            "title": res['title'],
            "response": res.get('response', res.get('error'))
        })

    df = pd.DataFrame(data)
    output_file = f"/data/image8/{tcno}.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Responses for {tcno} saved to {output_file}")

def process_folder(tcno):
    try:
        folder_path = f"/data/QAAPI/ASTRA_txt/{tcno}"
        all_docs = load_text_files_from_directory(folder_path)
        knowledge_base = process_text(all_docs)

        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_query, query, knowledge_base) for query in query_tittles.keys()]
            results = [future.result() for future in futures]

        save_tcno_to_excel(tcno, results)

    except Exception as e:
        print(f"Failed to process folder {tcno}: {str(e)}")

def process_folders_in_parallel():
    base_folder_path = r"/data/QAAPI/ASTRA_txt"
    tcno_folders = [tcno for tcno in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, tcno))]

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(process_folder, tcno_folders)

if __name__ == '__main__':
    process_folders_in_parallel()
