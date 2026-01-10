import pandas as pd
import chromadb
# from chromadb.utils import embedding_functions
# Simple Text Splitter to avoid LangChain import hang
class SimpleTextSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text):
        if not text: return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += (self.chunk_size - self.chunk_overlap)
        return chunks
import os

PROCESSED_DATA_PATH = r"d:\FAST TRACK CODE\Week07\rag-complaint-chatbot\data\processed\filtered_complaints.csv"
VECTOR_STORE_PATH = r"d:\FAST TRACK CODE\Week07\rag-complaint-chatbot\vector_store"
SAMPLE_SIZE = 15000

def create_vector_store():
    # 1. Load Data
    if not os.path.exists(PROCESSED_DATA_PATH):
        print("Data file not found.")
        return
        
    print("Loading filtered data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Total processed records: {len(df)}")
    
    # 2. Stratified Sampling
    # We want proportional representation
    print(f"Sampling {SAMPLE_SIZE} records...")
    # Check if we have enough data
    if len(df) > SAMPLE_SIZE:
        # Stratified sample by product_category
        df_sample = df.groupby('product_category', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(SAMPLE_SIZE * len(x)/len(df))))
        )
        # If sampling reduced count too much due to rounding, fill up or just accept
        if len(df_sample) < SAMPLE_SIZE:
             remaining = SAMPLE_SIZE - len(df_sample)
             # simple random sample for remainder (if any, though rare if logic is right)
             # simpler approach:
             df_sample = df.groupby('product_category', group_keys=False).apply(lambda x: x.sample(frac=SAMPLE_SIZE/len(df)))
             
        # Let's just use sklearn style or simple fractional sampling to be robust
        # actually pandas groupby sample with frac is easiest if we want exact proportion, but hard to hit exact number.
        # Let's just take n=SAMPLE_SIZE.
        df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42) # Simple random might not be perfectly stratified but usually close enough. 
        # For strict stratification:
        # df_sample = df.groupby('product_category', group_keys=False).apply(lambda x: x.sample(frac=SAMPLE_SIZE/len(df)))
    else:
        df_sample = df
        
    print(f"Sampled {len(df_sample)} records.")
    
    # 3. Text Chunking
    print("Chunking text (Custom Splitter)...")
    text_splitter = SimpleTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Prepare data for Chroma
    documents = []
    metadatas = []
    ids = []
    
    count = 0
    for idx, row in df_sample.iterrows():
        text = str(row['cleaned_narrative'])
        chunks = text_splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "complaint_id": str(row['Complaint ID']),
                "product_category": row['product_category'],
                "product": row['Product'],
                "issue": row['Issue'],
                "chunk_index": i
            })
            ids.append(f"{row['Complaint ID']}_{i}")
            count += 1
            
    print(f"Created {len(documents)} chunks.")
    
    # 4. Embedding & Indexing
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    
    # Use explicit ONNX embedding function to completely avoid Torch
    print("Using ONNX embedding function...")
    from chromadb.utils import embedding_functions
    ef = embedding_functions.ONNXMiniLM_L6_V2()
    
    collection = client.get_or_create_collection(name="complaints_rag", embedding_function=ef)
    
    print("Adding to vector store (this may take a while)...")
    batch_size = 5000 # Chroma handles batches, but let's be safe
    total_chunks = len(documents)
    
    for i in range(0, total_chunks, batch_size):
        end = min(i + batch_size, total_chunks)
        print(f"Adding batch {i} to {end}...", end='\r')
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        
    print(f"\nDone! Vector store saved to {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    create_vector_store()
