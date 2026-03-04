import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configurations import config
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
 
def ingest_pdfs():
    # 1. Find all PDFs in the current directory
    pdf_files = glob.glob("*.pdf")
    
    if not pdf_files:
        print("❌ No PDF files found in this folder.")
        return

    print(f"📂 Found {len(pdf_files)} PDFs: {pdf_files}")

    all_splits = []

    # 2. Loop through each PDF and load it
    for pdf_file in pdf_files:
        print(f"   Processing: {pdf_file}...")
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
            print(f"   ✅ Loaded {len(splits)} chunks from {pdf_file}")
            
        except Exception as e:
            print(f"   ⚠️ Error loading {pdf_file}: {e}")

    # 3. Create/Update the Vector Database
    if all_splits:
        print("\n🧠 Generating embeddings and saving to database... (This may take a moment)")
        
        # This will create the DB if missing, or add to it if it exists
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=OllamaEmbeddings(model=config.MODEL_NAME, base_url=config.OLLAMA_BASE_URL),
            persist_directory=config.PERSIST_DIRECTORY
        )
        
        print(f"🎉 Success! Database saved to '{config.PERSIST_DIRECTORY}'")
    else:
        print("❌ No valid text data found to process.")

if __name__ == "__main__":
    ingest_pdfs()