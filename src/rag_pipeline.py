import chromadb
from chromadb.utils import embedding_functions
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
import os

VECTOR_STORE_PATH = r"d:\FAST TRACK CODE\Week07\rag-complaint-chatbot\vector_store"
MODEL_ID = "google/flan-t5-small"

class ComplaintRAG:
    def __init__(self):
        print("Initializing RAG Pipeline (No LangChain)...")
        
        # 1. Load Vector Store
        print("Loading Vector Store...")
        # Use ONNX embedding function to match creation
        ef = embedding_functions.ONNXMiniLM_L6_V2()
        
        self.client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
        self.collection = self.client.get_collection(name="complaints_rag", embedding_function=ef)
        
        # 2. Load LLM
        if TRANSFORMERS_AVAILABLE:
            print(f"Loading LLM ({MODEL_ID})...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
                self.pipe = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=512,
                    truncation=True
                )
                self.has_llm = True
            except Exception as e:
                print(f"Failed to load local LLM: {e}")
                self.has_llm = False
        else:
            print("Transformers/Torch not available. RAG will be retrieval-only.")
            self.has_llm = False

    def answer_question(self, query):
        print(f"Querying: {query}")
        
        # Retrieve
        results = self.collection.query(
            query_texts=[query],
            n_results=5
        )
        
        # Parse results
        # Chroma returns lists of lists
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        context_parts = []
        source_documents = []
        
        for doc, meta in zip(docs, metadatas):
            context_parts.append(doc)
            # Create a mock object to match previous interface if needed, or just dict
            # Let's return dicts
            source_documents.append({'page_content': doc, 'metadata': meta})
            
        context = "\n\n".join(context_parts)
        
        answer = "LLM not available."
        
        if self.has_llm:
            prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            try:
                gen = self.pipe(prompt)
                answer = gen[0]['generated_text']
            except Exception as e:
                answer = f"Error generating answer: {e}"
        else:
            answer = "Error: PyTorch/Transformers not installed. Cannot generate answer. context retrieved."

        return {
            'result': answer,
            'source_documents': source_documents
        }

if __name__ == "__main__":
    # Test run
    rag = ComplaintRAG()
    response = rag.answer_question("What are the common issues with Credit Cards?")
    print("\nAnswer:", response['result'])
    print("\nSources:")
    for doc in response['source_documents']:
        print(f"- {doc['metadata'].get('product', 'N/A')}: {doc['page_content'][:100]}...")
