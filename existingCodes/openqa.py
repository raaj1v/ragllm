import os
import warnings
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import torch

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
warnings.filterwarnings("ignore")

# Define Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize the embedding model and LLM model globally
embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
llm = ChatOpenAI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="FAKE",
    max_tokens=512,
    temperature=0.7
)

# Function to process text files
def process_text(text):
    text = "\n".join([doc.page_content for doc in text])
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=32,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
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

@socketio.on('process_query')
def handle_query(data):
    try:
        tcno = data.get('tcno')
        query = data.get('query')

        if not tcno or not query:
            emit('error', {'error': 'Missing tcno or query'})
            return
        
        folder_path = f"/data/tendergpt/livetender_txt/{tcno}"
        all_docs = load_text_files_from_directory(folder_path)
        knowledge_base = process_text(all_docs)
        
        torch.cuda.empty_cache()

        # Process the query and stream the response in real-time
        process_query_simulated_streaming(query, knowledge_base)

        torch.cuda.empty_cache()

    except Exception as e:
        emit('error', {'error': str(e)})

def process_query_simulated_streaming(query, knowledge_base):
    try:
        docs = knowledge_base.similarity_search(query)
        
        # Combine the context from documents into a single string
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Question: {query}\nContext: {context}"
        
        # Generate the full response
        response = llm(prompt)
        
        # Split the response into chunks for simulated streaming
        chunk_size = 1000  # Adjust the chunk size as needed
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            emit('response_chunk', {'response': chunk})  # Emit each chunk as it's generated
        
        emit('response_chunk', {'response': '\n--- END OF RESPONSE ---\n'})  # Indicate the end of the response

    except Exception as e:
        emit('error', {'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
