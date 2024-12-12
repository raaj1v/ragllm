from datetime import timedelta, datetime
query_tittles = {
    "What are the functional requirements, also known as the scope of work, mentioned in the document?": "Scope of Work",
    "Clauses specifying  Pre-Qualification Criteria  or eligibility criteria": "Prequalification Criteria",
    "List all mandatory qualification criteria, including Blacklisting and required certifications": "Mandatory Qualification Criteria",
     "Performance criteria including work experience,experience and past performance criteria, emphasizing the need for prior similar project experience, references, and the successful completion of similar contracts": "Performance Criteria",
    "Financial criteria including turnover, Networth": "Financial Criteria",
    "Technical requirements": "Technical Requirements",
     "Work Specifications that bidders must meet to deliver tender requirements": "Specifications",
     "Supporting documents": "Supporting Documents",
    "Extract a comprehensive list of all dates, times, and monetary values, along with their specific labels or descriptions as mentioned in the document. This includes but is not limited to the following fields: bid submission end date, tender due date, bid validity, opening date, closing date, pre-bid meeting date, EMD date, tender value, and tender fee. Group all extracted items under the label 'Important Dates and Amounts,' clearly specifying each date, time, or amount and its description as stated in the document.":"Important date",
    "Extract the contact details, including phone number, email address, and officer name. If the details are unavailable, return 'None' for the missing fields.":"Contact details"      
}
import os
import torch
import pyodbc
import warnings
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from opensearchpy import OpenSearch as OpenSearchClient
from opensearchpy import OpenSearch
from elasticsearch import Elasticsearch
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from datetime import datetime
# # Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
warnings.filterwarnings("ignore")
# Global initialization of models
embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
# Load the embeddings model globally on CPU
llm_model = ChatOpenAI(
    # model_name="meta-llama/Meta-Llama-3-8B-Instruct",
     model_name="meta-llama/Llama-3.1-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="FAKE",
    max_tokens=4096,
    temperature=0.1
)
# Set up OpenSearch and Elasticsearch clients
index_name = 'tprocanswers'
opensearch_client = OpenSearchClient(
    hosts=['https://localhost:9200'],
    http_auth=("admin", "4Z*lwtz,,2T:0TGu"),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False
)
es_client = Elasticsearch(['http://141.148.218.254:9200'])
# Ensure OpenSearch index exists
if not opensearch_client.indices.exists(index=index_name):
    opensearch_client.indices.create(index=index_name)
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
def update_elasticsearch(tcno):
    search_query = {"query": {"term": {"tcno": tcno}}}
    indices = ['tbl_tendersadvance_migration_embedding', 'tbl_tendersadvance_migration']

    for index in indices:
        es_response = es_client.search(index=index, body=search_query)
        if es_response['hits']['total']['value'] > 0:
            for hit in es_response['hits']['hits']:
                es_client.update(index=index, id=hit['_id'], body={"doc": {"ispq": 1}})
                print(f"Updated tcno {tcno} with ispq = 1 in {index}")
def update_database(tcno):
    conn_string = (
        'DRIVER=/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.4.so.1.1;'
        'SERVER=140.238.241.137;DATABASE=ttneo;UID=aimlpq;PWD=aimlpq;'
        'TrustServerCertificate=yes;'
    )
    with pyodbc.connect(conn_string) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE apptender.tbl_tender SET ispq = 1 WHERE tcno = ?", (tcno,))
        conn.commit()
        print(f"Database updated successfully for tcno {tcno}")
def process_folder(tcno):
    try:
        # current_date = datetime.now().strftime("%d")
        folder_path = f"/data/txtfolder/dailydocument_06-12-24_txt/{tcno}"
        all_docs = load_text_files_from_directory(folder_path)
        knowledge_base = process_text(all_docs)

        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_query, query, knowledge_base) for query in query_tittles.keys()]
            results = [future.result() for future in futures]

        json_response = {
            "results": [{"title": result["title"], "response": result.get("response", result.get("error"))} for result in results]
        }

        opensearch_client.index(index=index_name, id=tcno, body=json_response)
        print(f"Indexed results for {tcno} in OpenSearch.")

        update_elasticsearch(tcno)
        update_database(tcno)

    except Exception as e:
        print(f"Failed to process folder {tcno}: {str(e)}")
def process_folders_in_parallel():
    # current_day = datetime.now().strftime("%d-%m-%y")
    base_folder_path = f"/data/txtfolder/dailydocument_06-12-24_txt"
    tcno_folders = [tcno for tcno in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, tcno))]

    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(process_folder, tcno_folders)
if __name__ == '__main__':
    process_folders_in_parallel()

