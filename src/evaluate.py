from src.rag_pipeline import ComplaintRAG
import pandas as pd
import time

def evaluate():
    print("Initialize System...")
    try:
        rag = ComplaintRAG()
    except Exception as e:
        print(f"Failed to init RAG: {e}")
        return

    questions = [
        "What are the common issues with Credit Cards?",
        "How do I dispute a transaction?",
        "Why was my account closed?",
        "issues with money transfers?",
        "Tell me about interest rates."
    ]
    
    results = []
    
    print("Running Evaluation...")
    for q in questions:
        start = time.time()
        print(f"Q: {q}")
        try:
            response = rag.answer_question(q)
            ans = response['result']
            sources = [doc['metadata'].get('product_category', 'Unknown') for doc in response['source_documents']]
            elapsed = time.time() - start
            
            results.append({
                "Question": q,
                "Answer": ans,
                "Sources": list(set(sources)), # Unique sources
                "Latency": f"{elapsed:.2f}s"
            })
            print(f"A: {ans[:100]}...\n")
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "Question": q,
                "Answer": f"Error: {e}",
                "Sources": [],
                "Latency": "N/A"
            })
            
    df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df)
    
    # Save to CSV for report
    df.to_csv("d:/FAST TRACK CODE/Week07/rag-complaint-chatbot/data/evaluation_results.csv", index=False)

if __name__ == "__main__":
    evaluate()
