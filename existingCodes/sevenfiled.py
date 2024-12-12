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




# embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
llm_model = ChatOpenAI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    # model_name ="meta-llama/Llama-3.2-3B",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="FAKE",
    max_tokens=128,
    temperature=0.1
)


# query_tittles = {
#     "Extract the Bid End Date/Time from the document. Provide only the bid end date and time without any extra details.": "bid_end_date",
#     "Extract the Ministry/State Name from the document. Provide only the ministry or state name without any extra details.": "ministry_state_name",
#     "Extract the Department Name/ from the document. Provide only the department name without any extra details.": "department_name",
#     "Extract the Organisation Name/ from the document. Provide only the organisation name without any extra details.": "organisation_name",
#     "Extract the Item Category/ from the document. Provide only the item category without any extra details.": "item_category",
#     "Extract the Bid Number/ from the document. Provide only the bid number without any extra details.": "bid_number",
#     "Extract the Contract Period/ from the document. Provide only the contract period without any extra details.": "contract_period",
#     "Extract the details of the Beneficiary/ from the document. Provide only the beneficiary details without any extra information.": "beneficiary_details",
#     "Extract the Type of Bid/ from the document. Provide only the type of bid without any extra details.": "bid_type",
#     "Extract the EMD Detail/ from the document. Provide the EMD details including the bank name.": "emd_detail",
#     "Extract the Estimated Bid Value/ from the document. Provide only the estimated bid value without any extra details.": "estimated_bid_value",
#     "Extract the EMD Exemption details/ from the document. Provide only the EMD exemption details without any extra information.": "emd_exemption"
# }

# # Query titles
# query_tittles = {
#     "Extract only the bid end date and time from the document. Provide only the bid date without any extra details.": "bid_date",
#     "Extract only the ministry or state name from the document. Provide only the ministry or state name without any extra details.":"ministry_state_name",
#     "Extract only the Department Name from the document. Provide only the Department Name without any extra details.":"Department_Name",
#     "Extract only the Organisation Name from the document. Provide only the Organisation Name without any extra details." :"Organisation_Name",
#     "Extract the list of all 'Item Category' sections from the tender document. Provide only the 'Item Category' text without any additional details.":"item_category",
#     "Extract the Bid Number from the tender document. Provide only the Bid Number without any extra details.":"bid_number",
#     "Extract only the contract period from the document. Provide only the contract period without any extra details." :"contract_period",
#     "Extract the details of the Beneficiary mentioned in the tender.":"Beneficiary_details",
#     "Extract only the type of bid from the document. Provide only the type of bid without any extra details.":"bid_type",
#     "Extract the EMD detail with the bank name mentioned in the tender." :"Emd_detail",
#     "Extract the estimated Bid Value   from the tender document. Provide only the estimated bid value without any extra details.":"Estimated_bid_value",
#     "Extract the full details of the EMD exemption from the tender document. Provide only the EMD exemption details without any extra information." :"EMD_EXEMPTION"
# }

# # Define individual queries for each point to ensure clarity
# query_tittles = {
#     "Extract the tender number, tender ID, or Tender Reference Number from the document?. (Provide only the number or ID without any additional context)": "Tender No",
#     "What is the name of the organization that issued the tender? (Provide only the organization name, with no extra context)": "Organization",
#     "Which state or region is this tender applicable to? (Provide only the state or region, no other details)": "State",
#     "What is the specific work location for this tender? (Provide only the work location)": "Work Location",
#     "Extract the closing date, submission deadline, due date, or Bid Submission End Date from the tender document. (Provide only the date and time, without any explanation or extra text)": "Closing Date",
#     "Extract the estimated tender value, Tender Value, contract value, or project cost. (Provide only the value, without any extra details or context)": "Tender Value",
#     "Extract the tender description, title, name of work, or project description. (Provide only the description or title, without any extra details or context)": "Short Description - Tender description",
#     "What is the EMD amount,Earnest Money Deposit or security deposit for this tender? (Provide only the EMD amount, without any extra context)": "EMD"
# }

# All params working fine except date and quantity 
# query_tittles = {
#     "Extract the tender number, tender ID, or Tender Reference Number from the document?. (Provide only the number or ID without any additional context)": "Tender No",
#     "What is the name of the organization that issued the tender? (Provide only the organization name, with no extra context)": "Organization",
#     "Which state or region is this tender applicable to? (Provide only the state or region, no other details)": "State",
#     "What is the specific work location for this tender? (Provide only the work location)": "Work Location",
#     "Extract the closing date, submission deadline, due date, or Bid Submission End Date from the tender document. (Provide only the date and time, without any explanation or extra text)": "Closing Date",
#     "Extract the estimated tender value, Tender Value, contract value, or project cost. (Provide only the value, without any extra details or context)": "Tender Value",
#     "Extract the tender description, title, name of work, or project description. (Provide only the description or title, without any extra details or context)": "Short Description - Tender description",
#     "What is the EMD amount, Earnest Money Deposit, or security deposit for this tender? (Provide only the EMD amount, without any extra context)": "EMD",
#     "Extract the URL of the tender or the e-procurement site. (Provide only the URL, without any extra context)": "URL",
#     "Extract the address provided in the tender document. (Provide only the address, without additional details)": "Address",
#     "Who is the officer or contact person listed for this tender? (Provide only the name, no other details)": "Officer Name",
#     "Extract the tender opening date. (Provide only the date, without extra details)": "Opening Date",
#     "Extract the tender publishing date. (Provide only the date, without any extra details)": "Publishing Date",
#     "Extract the pre-bid meeting date from the tender document. (Provide only the date, no other details)": "Pre-bid Date",
#     "Does the tender document mention corrigendum? (Answer Yes or No, without any extra details)": "Corrigendum",
#     "Is this an e-tender or e-auction? (Provide 'e-tender' or 'e-auction', without any extra details)": "E-tender/E-auction",
#     "Is the tender mentioned as NCB (National Competitive Bidding) or ICB (International Competitive Bidding)? (Provide either NCB or ICB, without any extra details)": "NCB/ICB",
#     "Extract the quantity or unit of measure specified in the tender document. (Provide only the quantity and unit, no extra details)": "Quantity-Unit",
#     "Extract the document fee mentioned in the tender. (Provide only the fee, without extra context)": "Document Fee"
# }

# the one which was in demo
# query_tittles = {
#     "Extract the tender number, tender ID, or Tender Reference Number from the document. (Provide only the number or ID without any additional context. If not available, return Refer to Tender Document.)": "Tender No",
#     "What is the name of the organization that issued the tender? (Provide only the organization name, with no extra context. If not available, return Refer to Tender Document.)": "Organization",
#     "Which state or region is this tender applicable to? (Provide only the state or region, no other details. If not available, return Refer to Tender Document.)": "State",
#     "What is the specific work location for this tender? (Provide only the work location. If not available, return Refer to Tender Document.)": "Work Location",
#     "Extract the closing date, submission deadline, due date, or Bid Submission End Date from the tender document. (Provide only the date and time, without any explanation or extra text. If not available, return Refer to Tender Document.)": "Closing Date",
#     "Extract the estimated tender value, Tender Value, contract value, or project cost. (Provide only the value, without any extra details or context. If not available, return Refer to Tender Document.)": "Tender Value",
#     "Extract the tender description, title, name of work, or project description. (Provide only the description or title, without any extra details or context. If not available, return Refer to Tender Document.)": "Short Description - Tender description",
#     "What is the EMD amount, Earnest Money Deposit, or security deposit for this tender? (Provide only the EMD amount, without any extra context. If not available, return Refer to Tender Document.)": "EMD",
#     "Extract the URL of the tender or the e-procurement site. (Provide only the URL, without any extra context. If not available, return Refer to Tender Document.)": "URL",
#     "Extract the address provided in the tender document. (Provide only the address, without additional details. If not available, return Refer to Tender Document.)": "Address",
#     "Who is the officer or contact person listed for this tender? (Provide only the name, no other details. If not available, return Refer to Tender Document.)": "Officer Name",
#     "Extract the tender opening date. (Provide only the date, without extra details. If not available, return Refer to Tender Document.)": "Opening Date",
#     "Extract the tender publishing date. (Provide only the date, without any extra details. If not available, return Refer to Tender Document.)": "Publishing Date",
#     "Extract the pre-bid meeting date from the tender document. (Provide only the date, no other details. If not available, return Refer to Tender Document.)": "Pre-bid Date",
#     "Does the tender document mention corrigendum? (Answer Yes or No, without any extra details. If not available, return Refer to Tender Document.)": "Corrigendum",
#     "Is this an e-tender or e-auction? (Provide 'e-tender' or 'e-auction', without any extra details. If not available, return Refer to Tender Document.)": "E-tender/E-auction",
#     "Is the tender mentioned as NCB (National Competitive Bidding) or ICB (International Competitive Bidding)? (Provide either NCB or ICB, without any extra details. If not available, return Refer to Tender Document.)": "NCB/ICB",
#     "Extract the quantity or unit of measure specified in the tender document. (Provide only the quantity and unit, no extra details. If not available, return Refer to Tender Document.)": "Quantity-Unit",
#     "Extract the document fee mentioned in the tender. (Provide only the fee, without extra context. If not available, return Refer to Tender Document.)": "Document Fee"
# }

query_tittles = {
    "Extract the tender number, tender ID, or Tender Reference Number from the document. (Provide only the number or ID without any additional context. If not available, return Refer to Tender Document.)": "Tender No",
    "What is the name of the organization that issued the tender? (Provide only the organization name, with no extra context. If not available, return Refer to Tender Document.)": "Organization",
    "What is the short name of the organization that issued the tender? (Provide only the short name. If not available, return Refer to Tender Document. Don't rewrite the organization name, if the shortname is present, you will see it)": "Organization Short Name",
    "What is the name of the department or ministry issuing this tender? (Provide only the name, no extra details. If not available, return Refer to Tender Document.)": "Department/Ministry",
    "What is the name of the division mentioned in the tender document? (Provide only the division name, no other details. If not available, return Refer to Tender Document.)": "Division",
    "Which city is this tender applicable to? (Provide only the city name, no other details. If not available, return Refer to Tender Document.)": "City",
    "Which state or region is this tender applicable to? (Provide only the state or region, no other details. If not available, return Refer to Tender Document.)": "State",
    "Which country is this tender applicable to? (Provide only the country name, no other details. If not available, return Refer to Tender Document.)": "Country",
    "What is the specific work location for this tender? (Provide only the work location. If not available, return Refer to Tender Document.)": "Work Location",
    "Extract the address provided in the tender document. (Provide only the address, without additional details. If not available, return Refer to Tender Document.)": "Address",
    "Extract the tender description, title, name of work, or project description. (Provide only the description or title, without any extra details or context. If not available, return Refer to Tender Document.)": "Short Description - Tender description",
    "Extract the quantity and unit of measurement specified in the tender document. (Provide only if linked to an item description. If not available, return Refer to Tender Document.)": "Quantity-Unit",
    "Extract the estimated tender value, Tender Value, contract value, or project cost. (Provide only the value, without any extra details or context. If not available, return Refer to Tender Document.)": "Tender Value",
    "What is the EMD amount, Earnest Money Deposit, or security deposit for this tender? (Provide only the EMD amount, without any extra context. If not available, return Refer to Tender Document.)": "EMD",
    "Extract the document fee mentioned in the tender. (Provide only the fee, without extra context. If not available, return Refer to Tender Document.)": "Document Fee",
    "Extract the URL of the tender or the e-procurement site. (Provide only the URL, without any extra context. If not available, return Refer to Tender Document.)": "URL",
    "Who is the officer or contact person listed for this tender? (Provide only the name, no other details. If not available, return Refer to Tender Document.)": "Officer Name",
    "Extract the tender publishing date and time. (Provide only the date and time, without any extra details. If not available, return Refer to Tender Document.)": "Publishing Date",
    "Extract the tender opening date and time. (Provide only the date and time, without extra details. If not available, return Refer to Tender Document.)": "Opening Date",
    "Extract the closing date and time, submission deadline, due date and time, or Bid Submission End Date  and time from the tender document. (Provide only the date and time, without any explanation or extra text. If not available, return Refer to Tender Document.)": "Closing Date",
    "Extract the pre-bid meeting date and time from the tender document. (Provide only the date and time, no other details. If not available, return Refer to Tender Document.)": "Pre-bid Date",
    "Extract the clarification start date and time from the tender document. (Provide only the date and time, no other details. If not available, return Refer to Tender Document.)": "Clarification Start Date",
    "Extract the clarification end date and time from the tender document. (Provide only the date and time, no other details. If not available, return Refer to Tender Document.)": "Clarification End Date",
    "Extract the document download start date and time from the tender document. (Provide only the date and time, no other details. If not available, return Refer to Tender Document.)": "Document Download start Date",
    "Extract the document download end date and time from the tender document. (Provide only the date and time, no other details. If not available, return Refer to Tender Document.)": "Document Download end Date",
    "Does the tender document mention corrigendum? (Answer Yes or No. If not available, return Refer to Tender Document.)": "Corrigendum",
    "Is this an online tender or an offline tender? (Provide only offline / online, if online, please write e-tender or e-auction, do not write unnecessary words.)": "Online/Offline",
    "Is this tender classified as NCB (National Competitive Bidding) or ICB (International Competitive Bidding)? (Provide NCB or ICB. If not available, return Refer to Tender Document.)": "NCB/ICB",
    "Extract the number of stages and their names. (Provide both, e.g., '2 stages, technical and financial'. If not available, return Refer to Tender Document.)": "Stages",
    "Does the tender mention any percentage? (Answer Yes or No. If not available, return Refer to Tender Document.)": "Percentage",
    "What is the contractor class mentioned in the tender? (Provide the class such as 'A', 'B', 'C',... or '1', '2', '3', '4',..... If not available, return Refer to Tender Document.)": "Contractor Class",
    "Extract the phone number from the tender document. (Provide only the number, no other details. If not available, return Refer to Tender Document.)": "Phone Number",
    "Extract the email address from the tender document. (Provide only the email address, without any additional details. If not available, return Refer to Tender Document.)": "Email ID"
}

# def process_text(text):
#     text = "\n".join([doc.page_content for doc in text])
#     text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2048, chunk_overlap=32)
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
        all_docs = load_text_files_from_directory(folder_path)
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
    output_file = f"/data/QAAPI/gem/{tcno}.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Responses for {tcno} saved to {output_file}")

def process_folder(tcno):
    try:
        folder_path = f"/data/tendergpt/Gem/{tcno}"
        knowledge_base = process_text(all_docs)

        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_query, query, knowledge_base) for query in query_tittles.keys()]
            results = [future.result() for future in futures]

        save_tcno_to_excel(tcno, results)

    except Exception as e:
        print(f"Failed to process folder {tcno}: {str(e)}")

def process_folders_in_parallel():
    base_folder_path = r"/data/tendergpt/Gem"
    tcno_folders = [tcno for tcno in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, tcno))]

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_folder, tcno_folders)

if __name__ == '__main__':
    process_folders_in_parallel()

