





## working with good result##


# import os
# import warnings
# import json
# import pickle
# import base64
# from typing import List

# import streamlit as st
# import requests

# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain_openai import ChatOpenAI
# from langchain.callbacks import get_openai_callback
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain_community.document_loaders import TextLoader, PyPDFLoader
# from langchain.embeddings.base import Embeddings

# # Suppress warnings
# warnings.filterwarnings("ignore")

# # Set environment variables (adjust as needed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

# # Initialize Streamlit app
# st.set_page_config(page_title="TenderGPT", layout="wide")
# st.title("TenderGPT - Document Query Interface")

# # Initialize LLM
# llm = ChatOpenAI(
#     model_name="meta-llama/Meta-Llama-3-8B-Instruct",
#     openai_api_base="http://localhost:8000/v1",
#     openai_api_key="FAKE",  # Replace with your actual API key
#     max_tokens=512,
#     temperature=0.1
# )

# # Define functions

# def get_embedding(text: str) -> List[float]:
#     """
#     Fetch embedding for the given text from the embedding API.
#     """
#     response = requests.post(
#         "http://0.0.0.0:5002/embeddings",
#         json={"model": "BAAI/bge-small-en-v1.5", "input": [text]}
#     )
#     if response.status_code == 200:
#         data = response.json()
#         return data['data'][0]['embedding']
#     else:
#         raise Exception(f"API request failed with status code {response.status_code}")

# class CustomEmbeddings(Embeddings):
#     """
#     Custom Embeddings class that utilizes an external API to generate embeddings.
#     """
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return [get_embedding(text) for text in texts]

#     def embed_query(self, text: str) -> List[float]:
#         return get_embedding(text)

# def process_text(texts):
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

# def save_embedding(tcno, knowledge_base):
#     """
#     Save the FAISS index to disk using pickle.
#     """
#     embeddings_dir = 'embeddings/'
#     os.makedirs(embeddings_dir, exist_ok=True)  # Create directory if it doesn't exist
#     save_path = os.path.join(embeddings_dir, f"{tcno}.pkl")
#     with open(save_path, 'wb') as f:
#         pickle.dump(knowledge_base, f)
#     st.success(f"Embeddings saved for TCNO {tcno} at `{save_path}`")

# def load_embedding(tcno):
#     """
#     Load the FAISS index from disk if it exists.
#     """
#     load_path = os.path.join('embeddings/', f"{tcno}.pkl")
#     if os.path.exists(load_path):
#         with open(load_path, 'rb') as f:
#             knowledge_base = pickle.load(f)
#         st.info(f"Loaded embeddings for TCNO {tcno} from `{load_path}`")
#         return knowledge_base
#     else:
#         return None

# def load_text_files(uploaded_files, tcno):
#     """
#     Load and process uploaded text and PDF files.
#     Also, save uploaded PDFs to 'stored_pdfs/{tcno}/' for future display.
#     """
#     all_docs = []
#     temp_dir = "temp_files"
#     os.makedirs(temp_dir, exist_ok=True)

#     # Directory to store uploaded PDFs
#     stored_pdfs_dir = os.path.join("stored_pdfs", tcno)
#     os.makedirs(stored_pdfs_dir, exist_ok=True)

#     for uploaded_file in uploaded_files:
#         try:
#             file_extension = os.path.splitext(uploaded_file.name)[1].lower()
#             temp_path = os.path.join(temp_dir, uploaded_file.name)
            
#             with open(temp_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
            
#             if file_extension == ".txt":
#                 loader = TextLoader(temp_path)
#                 docs = loader.load()
#                 all_docs.extend(docs)
#                 st.info(f"Loaded text file: `{uploaded_file.name}`")
#             elif file_extension == ".pdf":
#                 loader = PyPDFLoader(temp_path)
#                 docs = loader.load()
#                 all_docs.extend(docs)
#                 st.info(f"Loaded PDF file: `{uploaded_file.name}`")
                
#                 # **New Modification: Clear existing PDFs before saving new ones**
#                 # This ensures that only the latest uploaded PDFs are stored and displayed
#                 clear_stored_pdfs(stored_pdfs_dir)
                
#                 # Save the uploaded PDF to the stored_pdfs directory
#                 stored_pdf_path = os.path.join(stored_pdfs_dir, uploaded_file.name)
#                 with open(stored_pdf_path, "wb") as f:
#                     f.write(uploaded_file.getbuffer())
                
#                 # Display PDF in the Streamlit app
#                 display_pdf(stored_pdf_path)

#             else:
#                 st.warning(f"Unsupported file type: `{uploaded_file.name}`")
#         except Exception as e:
#             st.error(f"Error loading `{uploaded_file.name}`: {e}")
#     return all_docs

# def display_pdf(pdf_path):
#     """
#     Display a PDF file in the Streamlit app using an iframe.
    
#     """
#     try:
#         with open(pdf_path, "rb") as f:
#             pdf_data = f.read()
#         b64_pdf = base64.b64encode(pdf_data).decode()
#         pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="1200" height="600" frameborder="0"></iframe>'
#         st.markdown(pdf_display, unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"Failed to display PDF `{pdf_path}`: {e}")

# def display_stored_pdfs(tcno):
#     """
#     Display all stored PDFs for a given TCNO.
#     """
#     stored_pdfs_dir = os.path.join("stored_pdfs", tcno)
#     if os.path.exists(stored_pdfs_dir):
#         pdf_files = [f for f in os.listdir(stored_pdfs_dir) if f.lower().endswith(".pdf")]
#         if pdf_files:
#             st.subheader("Stored PDFs:")
#             for pdf_file in pdf_files:
#                 pdf_path = os.path.join(stored_pdfs_dir, pdf_file)
#                 st.write(f"**{pdf_file}**")
#                 display_pdf(pdf_path)
#         else:
#             st.info("No stored PDFs found for this TCNO.")
#     else:
#         st.info("No stored PDFs found for this TCNO.")

# def clear_stored_pdfs(stored_pdfs_dir):
#     """
#     Delete all PDFs in the stored_pdfs_dir to ensure only new uploads are stored.
#     """
#     if os.path.exists(stored_pdfs_dir):
#         pdf_files = [f for f in os.listdir(stored_pdfs_dir) if f.lower().endswith(".pdf")]
#         for pdf_file in pdf_files:
#             pdf_path = os.path.join(stored_pdfs_dir, pdf_file)
#             try:
#                 os.remove(pdf_path)
#                 st.info(f"Removed previous PDF: `{pdf_file}`")
#             except Exception as e:
#                 st.warning(f"Failed to remove `{pdf_file}`: {e}")

# def process_query(query, knowledge_base):
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
#             st.write(f"**OpenAI Cost:** {cost}")

#         return response
#     except Exception as e:
#         st.error(f"Error processing query: {e}")
#         return None

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
#             except Exception as e:
#                 st.warning(f"Failed to delete `{file_path}`: {e}")

# # Initialize session state variables
# if 'knowledge_base' not in st.session_state:
#     st.session_state.knowledge_base = None
# if 'tcno' not in st.session_state:
#     st.session_state.tcno = ""
# if 'uploaded_files' not in st.session_state:
#     st.session_state.uploaded_files = []

# # Streamlit UI Components

# st.header("Step 1: Provide TCNO and Upload Documents")

# with st.form("upload_form"):
#     tcno_input = st.text_input("Enter TCNO:", value=st.session_state.tcno)
#     uploaded_files = st.file_uploader(
#         "Upload Text or PDF Files", 
#         type=["txt", "pdf"], 
#         accept_multiple_files=True
#     )
#     submit_upload = st.form_submit_button("Process Documents")

# if submit_upload:
#     if not tcno_input:
#         st.error("Please enter a TCNO.")
#     else:
#         st.session_state.tcno = tcno_input  # Update session state
#         if uploaded_files:
#             with st.spinner("Processing uploaded documents..."):
#                 # Load and process uploaded files
#                 all_docs = load_text_files(uploaded_files, tcno_input)
                
#                 if all_docs:
#                     # Load existing embeddings if available
#                     knowledge_base = load_embedding(tcno_input)
#                     if not knowledge_base:
#                         # Create new knowledge base
#                         knowledge_base = process_text(all_docs)
#                         save_embedding(tcno_input, knowledge_base)
#                         st.session_state.knowledge_base = knowledge_base  # Update session state
#                         cleanup_temp_files()  # Clean up temporary files
#                     else:
#                         # Update existing knowledge base with new documents
#                         knowledge_base.add_texts([doc.page_content for doc in all_docs])
#                         save_embedding(tcno_input, knowledge_base)
#                         st.session_state.knowledge_base = knowledge_base  # Update session state
#                         cleanup_temp_files()  # Clean up temporary files
#                     st.success("Documents processed and embeddings are ready.")
#                 else:
#                     st.error("No documents were loaded.")
#         else:
#             # No new files uploaded; display stored PDFs if any
#             st.info("No new files uploaded. Displaying previously uploaded PDFs if available.")
#             display_stored_pdfs(tcno_input)
            
#             # Attempt to load existing embeddings
#             knowledge_base = load_embedding(tcno_input)
#             if knowledge_base:
#                 st.session_state.knowledge_base = knowledge_base
#                 st.info("Using existing embeddings.")
#             else:
#                 st.warning("No existing embeddings found for this TCNO. Please upload documents to create embeddings.")

# # **New Addition: Always display stored PDFs if TCNO is provided**
# # This ensures that PDFs are visible regardless of other interactions like querying

# if st.session_state.tcno:
#     st.header("Stored PDFs")
#     display_stored_pdfs(st.session_state.tcno)

# st.header("Step 2: Query the Knowledge Base")

# if st.session_state.knowledge_base is not None:
#     with st.form("query_form"):
#         query = st.text_input("Enter your query:")
#         submit_query = st.form_submit_button("Get Response")
    
#     if submit_query:
#         if not query.strip():
#             st.error("Please enter a valid query.")
#         else:
#             with st.spinner("Processing query..."):
#                 response = process_query(query, st.session_state.knowledge_base)
#                 if response:
#                     st.subheader("Response:")
#                     st.write(response)
# else:
#     st.warning("Please upload and process documents first.")


# ## .doc,docx,.csv,.xlsx

# import os
# import subprocess
# import warnings
# import json
# import pickle
# import base64
# from typing import List

# import streamlit as st
# import requests
# import pandas as pd
# from docx import Document as DocxDocument
# from tabulate import tabulate

# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain_openai import ChatOpenAI
# from langchain.callbacks import get_openai_callback
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.schema import Document as LangchainDocument
# from langchain.embeddings.base import Embeddings
# from langchain_community.document_loaders import TextLoader, PyPDFLoader

# # Suppress warnings
# warnings.filterwarnings("ignore")

# # Set environment variables (adjust as needed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

# # Initialize Streamlit app
# st.set_page_config(page_title="TenderGPT", layout="wide")
# st.title("TenderGPT - Document Query Interface")

# # Initialize LLM
# llm = ChatOpenAI(
#     model_name="meta-llama/Meta-Llama-3-8B-Instruct",
#     openai_api_base="http://localhost:8000/v1",
#     openai_api_key="FAKE",  # Replace with your actual API key
#     max_tokens=512,
#     temperature=0.1
# )

# # Define functions

# def get_embedding(text: str) -> List[float]:
#     """
#     Fetch embedding for the given text from the embedding API.
#     """
#     response = requests.post(
#         "http://0.0.0.0:5002/embeddings",
#         json={"model": "BAAI/bge-small-en-v1.5", "input": [text]}
#     )
#     if response.status_code == 200:
#         data = response.json()
#         return data['data'][0]['embedding']
#     else:
#         raise Exception(f"API request failed with status code {response.status_code}")

# class CustomEmbeddings(Embeddings):
#     """
#     Custom Embeddings class that utilizes an external API to generate embeddings.
#     """
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return [get_embedding(text) for text in texts]

#     def embed_query(self, text: str) -> List[float]:
#         return get_embedding(text)

# def process_document(file_path):
#     """
#     Process .doc and .docx files to extract text and tables.
#     """
#     def convert_doc_to_txt(file_path, output_file):
#         # Convert .doc to .txt using LibreOffice
#         try:
#             subprocess.run(['libreoffice', '--headless', '--convert-to', 'txt:Text', '--outdir', os.path.dirname(output_file), file_path], check=True)
#             os.rename(file_path.replace('.doc', '.txt'), output_file)
#         except subprocess.CalledProcessError as e:
#             st.error(f"LibreOffice conversion failed: {e}")
#             raise e

#     def read_docx(file_path1):
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

#     def read_txt(file_path1):
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

# def process_text(texts):
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

# def save_embedding(tcno, knowledge_base):
#     """
#     Save the FAISS index to disk using pickle.
#     """
#     embeddings_dir = 'embeddings/'
#     os.makedirs(embeddings_dir, exist_ok=True)  # Create directory if it doesn't exist
#     save_path = os.path.join(embeddings_dir, f"{tcno}.pkl")
#     with open(save_path, 'wb') as f:
#         pickle.dump(knowledge_base, f)
#     st.success(f"Embeddings saved for TCNO {tcno} at `{save_path}`")

# def load_embedding(tcno):
#     """
#     Load the FAISS index from disk if it exists.
#     """
#     load_path = os.path.join('embeddings/', f"{tcno}.pkl")
#     if os.path.exists(load_path):
#         with open(load_path, 'rb') as f:
#             knowledge_base = pickle.load(f)
#         st.info(f"Loaded embeddings for TCNO {tcno} from `{load_path}`")
#         return knowledge_base
#     else:
#         return None

# def load_text_files(uploaded_files, tcno):
#     """
#     Load and process uploaded text, PDF, DOC, DOCX, CSV, and Excel files.
#     Also, save uploaded PDFs and Word documents to 'stored_files/{tcno}/' for future display.
#     """
#     all_docs = []
#     temp_dir = "temp_files"
#     os.makedirs(temp_dir, exist_ok=True)

#     # Directory to store uploaded PDFs and Word documents
#     stored_files_dir = os.path.join("stored_files", tcno)
#     os.makedirs(stored_files_dir, exist_ok=True)

#     for uploaded_file in uploaded_files:
#         try:
#             file_extension = os.path.splitext(uploaded_file.name)[1].lower()
#             temp_path = os.path.join(temp_dir, uploaded_file.name)
            
#             with open(temp_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
            
#             if file_extension == ".txt":
#                 loader = TextLoader(temp_path)
#                 docs = loader.load()
#                 all_docs.extend(docs)
#                 st.info(f"Loaded text file: `{uploaded_file.name}`")
            
#             elif file_extension == ".pdf":
#                 loader = PyPDFLoader(temp_path)
#                 docs = loader.load()
#                 all_docs.extend(docs)
#                 st.info(f"Loaded PDF file: `{uploaded_file.name}`")
                
#                 # Clear existing PDFs before saving new ones
#                 clear_stored_files(stored_files_dir, [".pdf"])
                
#                 stored_file_path = os.path.join(stored_files_dir, uploaded_file.name)
#                 with open(stored_file_path, "wb") as f:
#                     f.write(uploaded_file.getbuffer())
                
#                 display_pdf(stored_file_path)
            
#             elif file_extension in [".doc", ".docx"]:
#                 text, tables_text = process_document(temp_path)
#                 st.info(f"Loaded document file: `{uploaded_file.name}`")
#                 st.text_area(f"Content of {uploaded_file.name}", text, height=200)
#                 if tables_text:
#                     st.text_area(f"Tables in {uploaded_file.name}", tables_text, height=200)
                
#                 # Clear existing Word documents before saving new ones
#                 clear_stored_files(stored_files_dir, [".doc", ".docx"])
                
#                 # Save the uploaded Word document to the stored_files directory
#                 stored_file_path = os.path.join(stored_files_dir, uploaded_file.name)
#                 with open(stored_file_path, "wb") as f:
#                     f.write(uploaded_file.getbuffer())
                
#                 # Optionally, display the content
#                 # display_text_from_doc(stored_file_path)
                
#                 # Create Langchain Document for FAISS processing
#                 combined_text = text
#                 if tables_text:
#                     combined_text += "\n" + tables_text
#                 langchain_doc = LangchainDocument(page_content=combined_text)
#                 all_docs.append(langchain_doc)
            
#             elif file_extension == ".csv":
#                 df = pd.read_csv(temp_path)
#                 text = df.to_string(index=False)
#                 langchain_doc = LangchainDocument(page_content=text)
#                 all_docs.append(langchain_doc)
#                 st.info(f"Loaded CSV file: `{uploaded_file.name}`")
            
#             elif file_extension in [".xls", ".xlsx"]:
#                 df = pd.read_excel(temp_path)
#                 text = df.to_string(index=False)
#                 langchain_doc = LangchainDocument(page_content=text)
#                 all_docs.append(langchain_doc)
#                 st.info(f"Loaded Excel file: `{uploaded_file.name}`")
            
#             else:
#                 st.warning(f"Unsupported file type: `{uploaded_file.name}`")
#         except Exception as e:
#             st.error(f"Error loading `{uploaded_file.name}`: {e}")
#     return all_docs

# def display_pdf(pdf_path):
#     """
#     Display a PDF file in the Streamlit app using an iframe.
#     """
#     try:
#         with open(pdf_path, "rb") as f:
#             pdf_data = f.read()
#         b64_pdf = base64.b64encode(pdf_data).decode()
#         pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="1200" height="600" frameborder="0"></iframe>'
#         st.markdown(pdf_display, unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"Failed to display PDF `{pdf_path}`: {e}")

# def display_stored_files(tcno):
#     """
#     Display all stored PDFs and Word documents for a given TCNO.
#     """
#     stored_files_dir = os.path.join("stored_files", tcno)
#     if os.path.exists(stored_files_dir):
#         # Display PDFs
#         pdf_files = [f for f in os.listdir(stored_files_dir) if f.lower().endswith(".pdf")]
#         if pdf_files:
#             st.subheader("Stored PDFs:")
#             for pdf_file in pdf_files:
#                 pdf_path = os.path.join(stored_files_dir, pdf_file)
#                 st.write(f"**{pdf_file}**")
#                 display_pdf(pdf_path)
#         else:
#             st.info("No stored PDFs found for this TCNO.")
        
#         # Display Word Documents
#         word_files = [f for f in os.listdir(stored_files_dir) if f.lower().endswith((".doc", ".docx"))]
#         if word_files:
#             st.subheader("Stored Word Documents:")
#             for word_file in word_files:
#                 word_path = os.path.join(stored_files_dir, word_file)
#                 st.write(f"**{word_file}**")
#                 # Optionally, display content
#                 # display_text_from_doc(word_path)
#         else:
#             st.info("No stored Word documents found for this TCNO.")
        
#         # Display CSV Files
#         csv_files = [f for f in os.listdir(stored_files_dir) if f.lower().endswith(".csv")]
#         if csv_files:
#             st.subheader("Stored CSV Files:")
#             for csv_file in csv_files:
#                 csv_path = os.path.join(stored_files_dir, csv_file)
#                 try:
#                     df = pd.read_csv(csv_path)
#                     st.write(f"**{csv_file}**")
#                     st.dataframe(df)
#                 except Exception as e:
#                     st.error(f"Failed to display CSV `{csv_file}`: {e}")
#         else:
#             st.info("No stored CSV files found for this TCNO.")
        
#         # Display Excel Files
#         excel_files = [f for f in os.listdir(stored_files_dir) if f.lower().endswith((".xls", ".xlsx"))]
#         if excel_files:
#             st.subheader("Stored Excel Files:")
#             for excel_file in excel_files:
#                 excel_path = os.path.join(stored_files_dir, excel_file)
#                 try:
#                     df = pd.read_excel(excel_path)
#                     st.write(f"**{excel_file}**")
#                     st.dataframe(df)
#                 except Exception as e:
#                     st.error(f"Failed to display Excel `{excel_file}`: {e}")
#         else:
#             st.info("No stored Excel files found for this TCNO.")
#     else:
#         st.info("No stored files found for this TCNO.")

# def clear_stored_files(stored_files_dir, extensions):
#     """
#     Delete all files with specified extensions in the stored_files_dir to ensure only new uploads are stored.
#     """
#     if os.path.exists(stored_files_dir):
#         files = [f for f in os.listdir(stored_files_dir) if f.lower().endswith(tuple(extensions))]
#         for file in files:
#             file_path = os.path.join(stored_files_dir, file)
#             try:
#                 os.remove(file_path)
#                 st.info(f"Removed previous file: `{file}`")
#             except Exception as e:
#                 st.warning(f"Failed to remove `{file}`: {e}")

# def process_query(query, knowledge_base):
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
#             st.write(f"**OpenAI Cost:** {cost}")

#         return response
#     except Exception as e:
#         st.error(f"Error processing query: {e}")
#         return None

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
#             except Exception as e:
#                 st.warning(f"Failed to delete `{file_path}`: {e}")

# # Initialize session state variables
# if 'knowledge_base' not in st.session_state:
#     st.session_state.knowledge_base = None
# if 'tcno' not in st.session_state:
#     st.session_state.tcno = ""
# if 'uploaded_files' not in st.session_state:
#     st.session_state.uploaded_files = []

# # Streamlit UI Components

# st.header("Step 1: Provide TCNO and Upload Documents")

# with st.form("upload_form"):
#     tcno_input = st.text_input("Enter TCNO:", value=st.session_state.tcno)
#     uploaded_files = st.file_uploader(
#         "Upload Text, PDF, Word, CSV, or Excel Files", 
#         type=["txt", "pdf", "doc", "docx", "csv", "xls", "xlsx"], 
#         accept_multiple_files=True
#     )
#     submit_upload = st.form_submit_button("Process Documents")

# if submit_upload:
#     if not tcno_input:
#         st.error("Please enter a TCNO.")
#     else:
#         st.session_state.tcno = tcno_input  # Update session state
#         if uploaded_files:
#             with st.spinner("Processing uploaded documents..."):
#                 # Load and process uploaded files
#                 all_docs = load_text_files(uploaded_files, tcno_input)
                
#                 if all_docs:
#                     # Load existing embeddings if available
#                     knowledge_base = load_embedding(tcno_input)
#                     if not knowledge_base:
#                         # Create new knowledge base
#                         knowledge_base = process_text(all_docs)
#                         save_embedding(tcno_input, knowledge_base)
#                         st.session_state.knowledge_base = knowledge_base  # Update session state
#                         cleanup_temp_files()  # Clean up temporary files
#                     else:
#                         # Update existing knowledge base with new documents
#                         knowledge_base.add_texts([doc.page_content for doc in all_docs])
#                         save_embedding(tcno_input, knowledge_base)
#                         st.session_state.knowledge_base = knowledge_base  # Update session state
#                         cleanup_temp_files()  # Clean up temporary files
#                     st.success("Documents processed and embeddings are ready.")
#                 else:
#                     st.error("No documents were loaded.")
#         else:
#             # No new files uploaded; display stored files if any
#             st.info("No new files uploaded. Displaying previously uploaded files if available.")
#             display_stored_files(tcno_input)
            
#             # Attempt to load existing embeddings
#             knowledge_base = load_embedding(tcno_input)
#             if knowledge_base:
#                 st.session_state.knowledge_base = knowledge_base
#                 st.info("Using existing embeddings.")
#             else:
#                 st.warning("No existing embeddings found for this TCNO. Please upload documents to create embeddings.")

# # Always display stored files if TCNO is provided
# if st.session_state.tcno:
#     st.header("Stored Files")
#     display_stored_files(st.session_state.tcno)

# st.header("Step 2: Query the Knowledge Base")

# if st.session_state.knowledge_base is not None:
#     with st.form("query_form"):
#         query = st.text_input("Enter your query:")
#         submit_query = st.form_submit_button("Get Response")
    
#     if submit_query:
#         if not query.strip():
#             st.error("Please enter a valid query.")
#         else:
#             with st.spinner("Processing query..."):
#                 response = process_query(query, st.session_state.knowledge_base)
#                 if response:
#                     st.subheader("Response:")
#                     st.write(response)
# else:
#     st.warning("Please upload and process documents first.")



import sys, ast
import sys
sys.path.append(r'/data/imageExtraction/')
sys.path.append(r'/data/imageExtraction/imgEnv/')

try:
    from deep_translator import GoogleTranslator
except ModuleNotFoundError:
    print("deep_translator module not found. Please check installation.")

import os
import logging
import io
import subprocess
import warnings
import json
import pickle
import base64
from typing import List

import streamlit as st
import requests
import pandas as pd
# from docx import Document as DocxDocument
from tabulate import tabulate
from PyPDF2 import PdfReader
from google.oauth2 import service_account
from googleapiclient.discovery import build
from apiclient.http import MediaFileUpload, MediaIoBaseDownload
from deep_translator import GoogleTranslator
from langdetect import detect
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document as LangchainDocument
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Set environment variables (adjust as needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Streamlit app
st.set_page_config(page_title="TenderGPT", layout="wide")
st.title("TenderGPT - Document Query Interface")

# Initialize LLM
llm = ChatOpenAI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="FAKE",  # Replace with your actual API key
    max_tokens=1024,
    temperature=0.1
)


# Define functions

def get_embedding(text: str) -> List[float]:
    """
    Fetch embedding for the given text from the embedding API.
    """
    response = requests.post(
        "http://0.0.0.0:5002/embeddings",
        json={"model": "BAAI/bge-small-en-v1.5", "input": [text]}
    )
    if response.status_code == 200:
        data = response.json()
        return data['data'][0]['embedding']
    else:
        raise Exception(f"API request failed with status code {response.status_code}")

class CustomEmbeddings(Embeddings):
    """
    Custom Embeddings class that utilizes an external API to generate embeddings.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return get_embedding(text)

def process_text(texts):
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

def save_embedding(tcno, knowledge_base):
    """
    Save the FAISS index to disk using pickle.
    """
    embeddings_dir = 'embeddings/'
    os.makedirs(embeddings_dir, exist_ok=True)  # Create directory if it doesn't exist
    save_path = os.path.join(embeddings_dir, f"{tcno}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(knowledge_base, f)
    st.success(f"Embeddings saved for TCNO {tcno} at `{save_path}`")


# Function to perform OCR using Google Drive API
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

# Function to process document files
def process_document(file_path):
    def convert_doc_to_txt(file_path, output_file):
        try:
            subprocess.run(['libreoffice', '--headless', '--convert-to', 'txt:Text', '--outdir', os.path.dirname(output_file), file_path], check=True)
            os.rename(file_path.replace('.doc', '.txt'), output_file)
        except subprocess.CalledProcessError as e:
            st.error(f"LibreOffice conversion failed: {e}")
            raise e

    def read_docx(file_path1):
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

    def read_txt(file_path1):
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

# Function to load and process uploaded files
def load_text_files(uploaded_files, tcno):
    all_docs = []
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    stored_files_dir = os.path.join("stored_files", tcno)
    os.makedirs(stored_files_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if file_extension == ".txt":
                loader = TextLoader(temp_path)
                docs = loader.load()
                all_docs.extend(docs)
                st.info(f"Loaded text file: `{uploaded_file.name}`")
            
            elif file_extension == ".pdf":
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                all_docs.extend(docs)
                st.info(f"Loaded PDF file: `{uploaded_file.name}`")
                
                clear_stored_files(stored_files_dir, [".pdf"])
                
                stored_file_path = os.path.join(stored_files_dir, uploaded_file.name)
                with open(stored_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                display_pdf(stored_file_path)
            
            elif file_extension in [".doc", ".docx"]:
                text, tables_text = process_document(temp_path)
                st.info(f"Loaded document file: `{uploaded_file.name}`")
                st.text_area(f"Content of {uploaded_file.name}", text, height=200)
                if tables_text:
                    st.text_area(f"Tables in {uploaded_file.name}", tables_text, height=200)
                
                clear_stored_files(stored_files_dir, [".doc", ".docx"])
                
                stored_file_path = os.path.join(stored_files_dir, uploaded_file.name)
                with open(stored_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                combined_text = text
                if tables_text:
                    combined_text += "\n" + tables_text
                langchain_doc = LangchainDocument(page_content=combined_text)
                all_docs.append(langchain_doc)

            elif file_extension in [".csv"]:
                df = pd.read_csv(temp_path)
                text = df.to_string(index=False)
                langchain_doc = LangchainDocument(page_content=text)
                all_docs.append(langchain_doc)
                st.info(f"Loaded CSV file: `{uploaded_file.name}`")
            
            elif file_extension in [".xls", ".xlsx"]:
                df = pd.read_excel(temp_path)
                text = df.to_string(index=False)
                langchain_doc = LangchainDocument(page_content=text)
                all_docs.append(langchain_doc)
                st.info(f"Loaded Excel file: `{uploaded_file.name}`")
            
            elif file_extension in [".jpg", ".jpeg", ".png"]:
                extracted_text = process_image(temp_path)
                st.image(temp_path, caption=uploaded_file.name, use_column_width=True)
                st.info(f"Displayed image: `{uploaded_file.name}`")
                langchain_doc = LangchainDocument(page_content=extracted_text)
                all_docs.append(langchain_doc)

            else:
                st.warning(f"Unsupported file type: `{uploaded_file.name}`")
        except Exception as e:
            st.error(f"Error loading `{uploaded_file.name}`: {e}")
    return all_docs


def load_embedding(tcno):
    """
    Load the FAISS index from disk if it exists.
    """
    load_path = os.path.join('embeddings/', f"{tcno}.pkl")
    if os.path.exists(load_path):
        with open(load_path, 'rb') as f:
            knowledge_base = pickle.load(f)
        st.info(f"Loaded embeddings for TCNO {tcno} from `{load_path}`")
        return knowledge_base
    else:
        return None


def display_pdf(pdf_path):
    """
    Display a PDF file in the Streamlit app using an iframe.
    """
    try:
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        b64_pdf = base64.b64encode(pdf_data).decode()
        pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="1200" height="600" frameborder="0"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to display PDF `{pdf_path}`: {e}")

def display_stored_files(tcno):
    """
    Display all stored PDFs and Word documents for a given TCNO.
    """
    stored_files_dir = os.path.join("stored_files", tcno)
    if os.path.exists(stored_files_dir):
        # Display PDFs
        pdf_files = [f for f in os.listdir(stored_files_dir) if f.lower().endswith(".pdf")]
        if pdf_files:
            st.subheader("Stored PDFs:")
            for pdf_file in pdf_files:
                pdf_path = os.path.join(stored_files_dir, pdf_file)
                st.write(f"**{pdf_file}**")
                display_pdf(pdf_path)
        else:
            st.info("No stored PDFs found for this TCNO.")
        
        # Display Word Documents
        word_files = [f for f in os.listdir(stored_files_dir) if f.lower().endswith((".doc", ".docx"))]
        if word_files:
            st.subheader("Stored Word Documents:")
            for word_file in word_files:
                word_path = os.path.join(stored_files_dir, word_file)
                st.write(f"**{word_file}**")
                # Optionally, display content
                # display_text_from_doc(word_path)
        else:
            st.info("No stored Word documents found for this TCNO.")
        
        # Display CSV Files
        csv_files = [f for f in os.listdir(stored_files_dir) if f.lower().endswith(".csv")]
        if csv_files:
            st.subheader("Stored CSV Files:")
            for csv_file in csv_files:
                csv_path = os.path.join(stored_files_dir, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                    st.write(f"**{csv_file}**")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Failed to display CSV `{csv_file}`: {e}")
        else:
            st.info("No stored CSV files found for this TCNO.")
        
        # Display Excel Files
        excel_files = [f for f in os.listdir(stored_files_dir) if f.lower().endswith((".xls", ".xlsx"))]
        if excel_files:
            st.subheader("Stored Excel Files:")
            for excel_file in excel_files:
                excel_path = os.path.join(stored_files_dir, excel_file)
                try:
                    df = pd.read_excel(excel_path)
                    st.write(f"**{excel_file}**")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Failed to display Excel `{excel_file}`: {e}")
        else:
            st.info("No stored Excel files found for this TCNO.")
    else:
        st.info("No stored files found for this TCNO.")

def clear_stored_files(stored_files_dir, extensions):
    """
    Delete all files with specified extensions in the stored_files_dir to ensure only new uploads are stored.
    """
    if os.path.exists(stored_files_dir):
        files = [f for f in os.listdir(stored_files_dir) if f.lower().endswith(tuple(extensions))]
        for file in files:
            file_path = os.path.join(stored_files_dir, file)
            try:
                os.remove(file_path)
                st.info(f"Removed previous file: `{file}`")
            except Exception as e:
                st.warning(f"Failed to remove `{file}`: {e}")

def process_query(query, knowledge_base):
    """
    Process the user's query against the knowledge base and return the response.
    """
    try:
        docs = knowledge_base.similarity_search(query, k=10)
        chain = load_qa_chain(llm, chain_type='stuff')
        
        instruction = (
            "Extract all relevant fields from the provided document and "
            "answer the input query based only on the dataset. "
            "Do not include irrelevant information."
        )

        full_input = f"{instruction}\n\nQuery: {query}"

        with get_openai_callback() as cost:
            response = chain.run(input_documents=docs, question=full_input)
            st.write(f"**OpenAI Cost:** {cost}")

        return response
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return None

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
            except Exception as e:
                st.warning(f"Failed to delete `{file_path}`: {e}")

# Initialize session state variables
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None
if 'tcno' not in st.session_state:
    st.session_state.tcno = ""
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Streamlit UI Components

st.header("Step 1: Provide TCNO and Upload Documents")

with st.form("upload_form"):
    tcno_input = st.text_input("Enter TCNO:", value=st.session_state.tcno)
    uploaded_files = st.file_uploader(
        "Upload Text, PDF, Word, CSV, or Excel Files", 
        type=["txt", "pdf", "doc", "docx", "csv", "xls","xlsx","jpg","jpeg","png"], 
        accept_multiple_files=True
    )
    submit_upload = st.form_submit_button("Process Documents")

if submit_upload:
    if not tcno_input:
        st.error("Please enter a TCNO.")
    else:
        st.session_state.tcno = tcno_input  # Update session state
        if uploaded_files:
            with st.spinner("Processing uploaded documents..."):
                # Load and process uploaded files
                all_docs = load_text_files(uploaded_files, tcno_input)
                
                if all_docs:
                    # Load existing embeddings if available
                    knowledge_base = load_embedding(tcno_input)
                    if not knowledge_base:
                        # Create new knowledge base
                        knowledge_base = process_text(all_docs)
                        save_embedding(tcno_input, knowledge_base)
                        st.session_state.knowledge_base = knowledge_base  # Update session state
                        cleanup_temp_files()  # Clean up temporary files
                    else:
                        # Update existing knowledge base with new documents
                        knowledge_base.add_texts([doc.page_content for doc in all_docs])
                        save_embedding(tcno_input, knowledge_base)
                        st.session_state.knowledge_base = knowledge_base  # Update session state
                        cleanup_temp_files()  # Clean up temporary files
                    st.success("Documents processed and embeddings are ready.")
                else:
                    st.error("No documents were loaded.")
        else:
            # No new files uploaded; display stored files if any
            st.info("No new files uploaded. Displaying previously uploaded files if available.")
            display_stored_files(tcno_input)
            
            # Attempt to load existing embeddings
            knowledge_base = load_embedding(tcno_input)
            if knowledge_base:
                st.session_state.knowledge_base = knowledge_base
                st.info("Using existing embeddings.")
            else:
                st.warning("No existing embeddings found for this TCNO. Please upload documents to create embeddings.")

# Always display stored files if TCNO is provided
if st.session_state.tcno:
    st.header("Stored Files")
    display_stored_files(st.session_state.tcno)

st.header("Step 2: Query the Knowledge Base")

if st.session_state.knowledge_base is not None:
    with st.form("query_form"):
        query = st.text_input("Enter your query:")
        submit_query = st.form_submit_button("Get Response")
    
    if submit_query:
        if not query.strip():
            st.error("Please enter a valid query.")
        else:
            with st.spinner("Processing query..."):
                response = process_query(query, st.session_state.knowledge_base)
                if response:
                    st.subheader("Response:")
                    st.write(response)
else:
    st.warning("Please upload and process documents first.")
