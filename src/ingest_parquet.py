import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import numpy as np

PARQUET_PATH = r"d:\FAST TRACK CODE\Week07\rag-complaint-chatbot\vector_store\complaint_embeddings.parquet"
VECTOR_STORE_PATH = r"d:\FAST TRACK CODE\Week07\rag-complaint-chatbot\vector_store"

def ingest_parquet():
    if not os.path.exists(PARQUET_PATH):
        print(f"Error: Parquet file not found at {PARQUET_PATH}")
        return

    print(f"Loading Parquet file: {PARQUET_PATH}")
    try:
        df = pd.read_parquet(PARQUET_PATH)
    except Exception as e:
        print(f"Error reading parquet: {e}")
        return

    print("Columns found:", df.columns.tolist())
    print("Sample row:", df.iloc[0].to_dict())
    
    # Initialize Chroma
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    
    # We need to know which column is the embedding
    # Usually it's 'embedding' or 'embeddings'
    embedding_col = None
    for col in ['embedding', 'embeddings', 'vector']:
        if col in df.columns:
            embedding_col = col
            break
    
    if not embedding_col:
        print("Error: Could not find embedding column in parquet.")
        return

    # Create or get collection
    # Note: If the parquet has embeddings, we might not need an embedding function for ingestion,
    # but we DO need it for retrieval later. 
    # Use ONNX to match the plan.
    ef = embedding_functions.ONNXMiniLM_L6_V2()
    collection = client.get_or_create_collection(name="complaints_rag", embedding_function=ef)

    print(f"Ingesting {len(df)} records into Chroma...")
    
    batch_size = 2000
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Prepare data
        documents = batch['Consumer complaint narrative'].astype(str).tolist() if 'Consumer complaint narrative' in batch.columns else batch['text'].astype(str).tolist()
        
        # Embeddings usually need to be a list of lists (floats)
        embeddings = batch[embedding_col].tolist()
        if isinstance(embeddings[0], np.ndarray):
            embeddings = [e.tolist() for e in embeddings]
        
        # IDs must be strings
        ids = batch['id'].astype(str).tolist() if 'id' in batch.columns else [str(uuid.uuid4()) for _ in range(len(batch))]
        if 'Complaint ID' in batch.columns:
            ids = batch['Complaint ID'].astype(str).tolist()

        # Metadata
        # Filter columns to only keep relevant metadata
        # Exclude text and embedding columns
        exclude_cols = [embedding_col, 'Consumer complaint narrative', 'text']
        metadata_cols = [c for c in batch.columns if c not in exclude_cols]
        metadatas = batch[metadata_cols].to_dict('records')
        # Ensure all metadata values are strings, ints, floats or bools
        for meta in metadatas:
            for k, v in meta.items():
                if not isinstance(v, (str, int, float, bool)):
                    meta[k] = str(v)

        print(f"Adding batch {i} to {i+len(batch)}...")
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    print("Ingestion complete!")

if __name__ == "__main__":
    import uuid
    ingest_parquet()
