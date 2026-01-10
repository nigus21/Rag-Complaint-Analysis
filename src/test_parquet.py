import pandas as pd
import os
path = r"d:\FAST TRACK CODE\Week07\rag-complaint-chatbot\vector_store\complaint_embeddings.parquet"
print(f"Checking {path}")
if os.path.exists(path):
    print("Found! Reading head...")
    try:
        df = pd.read_parquet(path, engine='auto')
        print("Columns:", df.columns.tolist())
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")
else:
    print("Not found")
