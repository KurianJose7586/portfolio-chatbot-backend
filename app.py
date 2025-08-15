import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import re


load_dotenv()

app = Flask(__name__)
CORS(app)

rag_chain = None

def json_to_text(json_data):
    """
    A simple function to convert the structured JSON into a single string.
    This text will be used for creating embeddings.
    """
    text = ""
    for key, value in json_data.items():
        if isinstance(value, dict):
            text += f"{key.replace('_', ' ').title()}:\n"
            for sub_key, sub_value in value.items():
                text += f"  {sub_key.replace('_', ' ').title()}: {sub_value}\n"
        elif isinstance(value, list):
            text += f"{key.replace('_', ' ').title()}:\n"
            for item in value:
                if isinstance(item, dict):
                    for item_key, item_value in item.items():
                        text += f"  - {item_key.replace('_', ' ').title()}: {item_value}\n"
                else:
                    text += f"  - {item}\n"
        else:
            text += f"{key.replace('_', ' ').title()}: {value}\n"
        text += "\n"
    return text

def initialize_rag_chain():
    global rag_chain
    try:
        # 1. Load data from JSON file
        print("Loading data from knowledge_base.json...")
        data_dir = "data"
        json_path = os.path.join(data_dir, 'knowledge_base.json')
        
        with open(json_path, 'r') as f:
            knowledge_base = json.load(f)
        
        # Convert the entire JSON to a single text string
        text_content = json_to_text(knowledge_base)
        # Wrap it in a LangChain Document object
        documents = [Document(page_content=text_content)]
        
        # 2. Chunk the documents
        print("Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks.")

        # 3. Create Embeddings
        print("Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        # 4. Create FAISS Vector Store
        print("Creating FAISS vector store...")
        vector_store = FAISS.from_documents(chunks, embedding_model)
        retriever = vector_store.as_retriever()

        # 5. Initialize LLM
        print("Initializing Groq LLM...")
        llm = ChatGroq(temperature=0, model_name="qwen/qwen3-32b")

        # 6. Create RAG Chain
        print("Creating RAG chain...")
        prompt = ChatPromptTemplate.from_template("""
        You are "KurianGPT", an expert and very friendly AI assistant providing information about Kurian Jose based on his resume and project documents.
        Answer the user's question based only on the following context.
        Refer too yourself as "KurianGPT" and Kurian as "Kurian".
        Avoid showing internal reasoning. Do not output <think> tags. Respond directly and professionally.
        Keep responses **very brief** (1–2 sentences max) unless the user asks for more details or examples.
        If the answer is not in the context, politely say that you can only answer questions regarding kurian's professional background and projects.
        refer to the user as "you" and Kurian as "Kurian".
        <context>
        {context}
        </context>

        Question: {input}
        """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        print("--- RAG Chain Initialized Successfully! ---")
    except Exception as e:
        print(f"Error during RAG initialization: {e}")

def strip_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

#App routes
@app.route('/api/chat', methods=['POST'])
def chat():
    if not rag_chain:
        return jsonify({'error': 'RAG chain is not initialized. Check server logs.'}), 500
    data = request.get_json()
    user_message = data.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    try:
        result = rag_chain.invoke({"input": user_message})
        raw_response = result.get('answer', "I couldn't generate a response.")
        clean_response = strip_think_tags(raw_response)
        return jsonify({'reply': clean_response})
    except Exception as e:
        print(f"Error during chat processing: {e}")
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

# Initialize the RAG chain when the application starts
initialize_rag_chain()

# This part is for local development only and will not be used by Vercel
if __name__ == '__main__':
    app.run(debug=True, port=5001)