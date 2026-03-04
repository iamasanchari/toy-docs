import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configurations import config
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- YOUR SPECIFIC IMPORTS ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


def start_chat():
    # 1. Check if Database exists
    if not os.path.exists(config.PERSIST_DIRECTORY):
        print("❌ No database found! Please run 'ingest.py' first to process your PDFs.")
        return

    print("🔄 Loading knowledge base...")
    
    # 2. Load the Database (No need to process PDFs again)
    vectorstore = Chroma(
        persist_directory=config.PERSIST_DIRECTORY,
        embedding_function=OllamaEmbeddings(model=config.MODEL_NAME, base_url=config.OLLAMA_BASE_URL)
    )
    
    # 3. Initialize the Brain
    llm = ChatOllama(model=config.MODEL_NAME, base_url=config.OLLAMA_BASE_URL)
    
    # 4. Setup the Prompt
    system_prompt = (
        "You are a helpful assistant answering questions based on the provided documents. "
        "Use the context below to answer accurately. "
        "If the answer is not in the context, say you don't know."
        "\n\n"
        "{context}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. Build the Chain
    retriever = vectorstore.as_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n✅ System Ready! (Type 'exit' to quit)")
    print("-" * 50)

    # 6. The Chat Loop
    while True:
        try:
            user_input = input("\n🗣️  You: ")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
            
            if not user_input.strip():
                continue

            print("🤖 Thinking...", end="\r")
            
            # Get response
            response = rag_chain.invoke({"input": user_input})
            
            # Print result
            print(f"🤖 AI: {response['answer']}")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_chat()