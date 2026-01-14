from src.config import SCRIPTS_DIR, DATA_OUTPUT_PATH
from src.processor import process_directory, save_dialogues
from src.vector_store import create_index_semantic  # or create_index_recursive

def main():
    print(f"Processing scripts from {SCRIPTS_DIR}...")
    # 1. Extract dialogues
    dialogues = process_directory(SCRIPTS_DIR, 'DATA')
    print(f"Extracted {len(dialogues)} lines.")
    
    # 2. Save processed data
    save_dialogues(dialogues, DATA_OUTPUT_PATH)
    print(f"Saved processed data to {DATA_OUTPUT_PATH}")
    
    # 3. Create Vector Store (Optional step here, or could be separate)
    # Reading back the file to ensure we process exactly what was saved
    with open(DATA_OUTPUT_PATH, encoding="utf-8") as f:
        data_lines = f.read()

    print("Creating vector index...")
    # Using semantic chunking as per original script
    # Note: Original script had recursive splitting AFTER semantic? 
    # Let's check the original script logic.
    # Original: 
    # 1. SemanticChunker to create docs (lines 86-92 in original file shown in Step 151 actually shows Recursive replacing Semantic??)
    # Wait, in Step 151, lines 77-80 create SemanticChunker, then line 86 reassigns text_splitter to RecursiveCharacterTextSplitter.
    # The 'docs' are created using the Recursive splitter (line 92).
    # So the SemanticChunker was unused code in the final version of the user's file.
    
    # I should check src/vector_store.py again. I implemented both.
    # I'll use Recursive to match the effective logic of the user's script.
    
    from src.vector_store import create_index_recursive, save_index
    
    vector_store = create_index_recursive(data_lines)
    
    # Save index locally for main.py to use
    save_index(vector_store, "faiss_index")
    print("Vector index created and saved to 'faiss_index'.")

if __name__ == "__main__":
    main()
