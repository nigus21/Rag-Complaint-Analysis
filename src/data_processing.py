import pandas as pd
import os
import re

RAW_DATA_PATH = r"d:\FAST TRACK CODE\Week07\rag-complaint-chatbot\data\raw\complaints.csv"
PROCESSED_DATA_PATH = r"d:\FAST TRACK CODE\Week07\rag-complaint-chatbot\data\processed\filtered_complaints.csv"

# Target Product Categories mapping
# We want: Credit Cards, Personal Loans, Savings Accounts, Money Transfers
# We will map CFPB categories to these standard labels.
PRODUCT_MAPPING = {
    'Credit card': 'Credit card',
    'Credit card or prepaid card': 'Credit card',
    'Prepaid card': 'Credit card', # ambiguous but maybe include? Challenge says "Credit Cards"
    
    'Checking or savings account': 'Savings account',
    'Savings account': 'Savings account', # older data might have this
    'Bank account or service': 'Savings account',
    
    'Payday loan, title loan, or personal loan': 'Personal loan',
    'Consumer Loan': 'Personal loan',
    'Payday loan': 'Personal loan',
    
    'Money transfer, virtual currency, or money service': 'Money transfers',
    'Money transfers': 'Money transfers'
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove simple boilerplate (this is hard without knowing exact boilerplate, but we can do basic cleaning)
    # Remove "xxxx" which are often redacted info in CFPB data
    text = re.sub(r'x{2,}', '', text)
    # Basic whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_data():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"File not found: {RAW_DATA_PATH}")
        return

    print("Filtering and processing data...")
    
    chunk_size = 100000
    header = True
    processed_count = 0
    
    # Delete existing file if any
    if os.path.exists(PROCESSED_DATA_PATH):
        os.remove(PROCESSED_DATA_PATH)

    for chunk in pd.read_csv(RAW_DATA_PATH, chunksize=chunk_size):
        # 1. Filter by Product
        # Normalize product names and check if in mapping
        mask = chunk['Product'].map(lambda x: x in PRODUCT_MAPPING or x in PRODUCT_MAPPING.keys())
        filtered_chunk = chunk[mask].copy()
        
        # Standardize Product names
        filtered_chunk['product_category'] = filtered_chunk['Product'].map(lambda x: PRODUCT_MAPPING.get(x, x))
        
        # 2. Filter empty narratives
        filtered_chunk = filtered_chunk.dropna(subset=['Consumer complaint narrative'])
        
        if filtered_chunk.empty:
            continue
            
        # 3. Clean text
        filtered_chunk['cleaned_narrative'] = filtered_chunk['Consumer complaint narrative'].apply(clean_text)
        
        # Save columns
        cols_to_save = ['Complaint ID', 'product_category', 'Product', 'Issue', 'Sub-issue', 'Company', 'State', 'Date received', 'cleaned_narrative']
        
        # Append to CSV
        mode = 'w' if header else 'a'
        filtered_chunk[cols_to_save].to_csv(PROCESSED_DATA_PATH, mode=mode, header=header, index=False)
        
        header = False
        processed_count += len(filtered_chunk)
        print(f"Processed {processed_count} valid records...", end='\r')
        
    print(f"\nCompleted! Saved {processed_count} records to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    process_data()
