from src.vector_store import load_index
from src.chatbot import build_rag_chain, query_chain
import os

def main():
    if not os.path.exists("faiss_index"):
        print("Index not found. Please run 'python ingest.py' first.")
        return

    print("Loading index...")
    vector_store = load_index("faiss_index")
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    print("Building brain...")
    chain = build_rag_chain(retriever)
    
    print("Ready! Ask Data a question (or type 'quit' to exit).")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        try:
            answer = query_chain(chain, user_input)
            print(f"Data: {answer}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
