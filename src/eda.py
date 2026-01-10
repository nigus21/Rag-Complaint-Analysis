import pandas as pd
import os

DATA_PATH = r"d:\FAST TRACK CODE\Week07\rag-complaint-chatbot\data\raw\complaints.csv"
OUTPUT_PATH = r"d:\FAST TRACK CODE\Week07\rag-complaint-chatbot\data\processed\filtered_complaints.csv"

def run_eda():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    print("Loading dataset (first 5 rows to inspect)...")
    df_head = pd.read_csv(DATA_PATH, nrows=5)
    print("Columns:", df_head.columns.tolist())
    
    use_cols = ['Product', 'Consumer complaint narrative', 'Complaint ID', 'Company', 'State', 'Date received', 'Issue', 'Sub-issue']
    
    print("Processing in chunks to find unique products and stats...")
    
    unique_products = set()
    total_records = 0
    non_empty_narratives = 0
    narrative_lengths = []
    
    chunk_size = 100000
    try:
        # First pass: Get unique products and simple stats for a sample (or full if fast enough)
        # We will read only the first few chunks to get an idea, or maybe all if we only aggregate.
        # Let's read all chunks but only keep aggregates.
        
        for chunk in pd.read_csv(DATA_PATH, usecols=use_cols, chunksize=chunk_size):
            unique_products.update(chunk['Product'].dropna().unique())
            total_records += len(chunk)
            
            # Count non-empty narratives
            valid_narratives = chunk['Consumer complaint narrative'].dropna()
            non_empty_narratives += len(valid_narratives)
            
            # Sample narrative lengths (first 1000 per chunk to save time)
            if len(valid_narratives) > 0:
                sample_lens = valid_narratives.head(1000).apply(lambda x: len(str(x).split()))
                narrative_lengths.extend(sample_lens.tolist())
                
            print(f"Processed {total_records} records...", end='\r')
            
            # If we just want to see unique products, we can stop early if we have a lot, 
            # but we want accurate counts for the 5 categories.
            # 6GB is ~10-20 chunks likely.
            
    except ValueError as e:
        print(f"Error loading columns: {e}")
        return

    print(f"\nTotal records: {total_records}")
    print("\n--- EDA ---")
    print("Unique Products:", sorted(list(unique_products)))
    print("Non-empty narratives:", non_empty_narratives)
    
    if narrative_lengths:
        s = pd.Series(narrative_lengths)
        print("\nNarrative Length Stats (Sample):")
        print(s.describe())

if __name__ == "__main__":
    run_eda()
