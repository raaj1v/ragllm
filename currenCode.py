
import re
import warnings
import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress warnings
warnings.filterwarnings("ignore")

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

class TenderAnalyzer:
    """Main class for analyzing tender documents"""
    
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
            "Extract clauses that specify Pre-Qualification Criteria or eligibility criteria while strictly avoiding duplicates in any points.": "Prequalification Criteria"
        }
        self.request_count = 0 
    def process_document(self, file_path: str) -> List[str]:
        """Process document and split into chunks"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        sentences = self._split_into_sentences(text)
        print("sentences", sentences)
        chunks = self._create_chunks(sentences)
        print("chunks", chunks)
        return self._chunk_by_tokens(chunks)

    def _split_into_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sentences with metadata"""
        sentences = [{'sentence': s, 'index': i} 
                    for i, s in enumerate(re.split(r'(?<=[.?!])\s+', text))]
        return self._combine_sentences(sentences)

    def _combine_sentences(self, sentences: List[Dict[str, Any]], buffer_size: int = 1) -> List[Dict[str, Any]]:
        """Combine sentences with context"""
        combined = []
        for i, sent in enumerate(sentences):
            context = []
            # Add previous sentences
            for j in range(max(0, i - buffer_size), i):
                context.append(sentences[j]['sentence'])
            # Add current and next sentences
            context.append(sent['sentence'])
            for j in range(i + 1, min(len(sentences), i + buffer_size + 1)):
                context.append(sentences[j]['sentence'])
            sent['combined_sentence'] = ' '.join(context)
            combined.append(sent)
        return combined

    def _create_chunks(self, sentences: List[Dict[str, Any]]) -> List[str]:
        """Create document chunks based on semantic similarity"""
        # Create embeddings
        embeddings = self.model.encode([s['combined_sentence'] for s in sentences])
        
        # Calculate distances
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]))
            distances.append(1 - similarity)
        
        # Split into chunks
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

    def _chunk_by_tokens(self, texts: List[str], max_tokens: int = 1000) -> List[str]:
        """Split texts into smaller chunks based on token count"""
        max_chars = max_tokens * 2
        chunks = []
        for text in texts:
            text_chunks = [text[i:i + max_chars] 
                         for i in range(0, len(text), max_chars)]
            chunks.extend(text_chunks)
        return chunks

    def process_query(self, query: str, text: str) -> str:
        """Process a single query against the text"""
        try:
            self.request_count += 1  # Increment the request counter
            
            # Print the current request details
            print(f"Request {self.request_count}:")
            print(f"Query: {query}")
            
            with get_openai_callback() as cb:
                response = self.chain.run(
                    input_documents=[Document(page_content=text)],
                    question=query
                )
            return response.strip()
        except Exception as e:
            print(f"Error processing query: {e}")
            return f"Error: {str(e)}"

    def analyze_tender(self, file_path: str) -> Dict[str, str]:
        """Main analysis function"""
        # Process document
        chunks = self.process_document(file_path)
        combined_text = " ".join(chunks)
        
        # Process queries in parallel
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

def analyze_tender_document(file_path: str) -> Dict[str, str]:
    """
    Top-level function to analyze a tender document
    
    Args:
        file_path (str): Path to the tender document
    
    Returns:
        Dict[str, str]: Dictionary of analysis results
    """
    analyzer = TenderAnalyzer()
    return analyzer.analyze_tender(file_path)

def main():
    """Main execution function"""
    # Process tender document
    input_file = "/data/Pqmatch/testing/78804029/78804029.txt"
    
    # Analyze and get results
    results = analyze_tender_document(input_file)
    
    # Print results (optional)
    import json
    print(json.dumps(results, indent=4))
    
    return results

if __name__ == "__main__":
    main()