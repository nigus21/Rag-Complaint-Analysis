import gradio as gr
from src.rag_pipeline import ComplaintRAG

# Initialize RAG at startup
rag = None

def get_rag():
    global rag
    if rag is None:
        try:
            rag = ComplaintRAG()
        except Exception as e:
            return None, f"Error initializing RAG: {str(e)}"
    return rag, "Ready"

def chat_function(message, history):
    rag_system, status = get_rag()
    if not rag_system:
        return f"System not ready: {status}"
    
    response = rag_system.answer_question(message)
    answer = response['result']
    documents = response['source_documents']
    
    source_text = "\n\n**Sources:**\n"
    for i, doc in enumerate(documents):
        product_name = doc['metadata'].get('product', 'Product') or 'Product'
        issue_name = doc['metadata'].get('issue', 'Issue') or 'Issue'
        content = doc.get('page_content', 'No content')[:200]
        source_text += f"{i+1}. **{product_name}** ({issue_name}): {content}...\n"
        
    return answer + source_text

# Create UI
with gr.Blocks(title="CrediTrust Complaint Assistant") as demo:
    gr.Markdown("# CrediTrust Complaint Assistant")
    gr.Markdown("Ask questions about customer complaints to get AI-synthesized insights.")
    
    chatbot = gr.ChatInterface(
        fn=chat_function,
        examples=["What are the main issues with Credit Cards?", "Why are customers complaining about Money Transfers?", "Tell me about fees."],
        title="Complaint Analysis Chatbot"
    )

if __name__ == "__main__":
    demo.launch()
