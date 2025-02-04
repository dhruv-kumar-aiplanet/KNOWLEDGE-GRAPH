import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from neo4j import GraphDatabase
import huggingface_hub

# Load environment variables
load_dotenv()

# Configure credentials
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Login to Hugging Face
huggingface_hub.login(token=HUGGINGFACE_TOKEN)

def init_llm():
    """Initialize Groq LLM"""
    return ChatGroq(
        temperature=0.7,
        model_name="llama2-70b-4096",
        groq_api_key=GROQ_API_KEY
    )

def init_embeddings():
    """Initialize HuggingFace embeddings with proper authentication"""
    model_kwargs = {
        'device': 'cpu',
        'token': HUGGINGFACE_TOKEN
    }
    
    encode_kwargs = {
        'normalize_embeddings': True
    }
    
    # Using a simpler, more reliable model
    return HuggingFaceEmbeddings(
        model_name="distilbert-base-uncased",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def test_neo4j_connection():
    """Test Neo4j connection and create constraints"""
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        
        with driver.session() as session:
            result = session.run("RETURN 1")
            result.single()
            
            session.run("""
                CREATE CONSTRAINT pdf_node_constraint IF NOT EXISTS
                FOR (n:PDFNode) REQUIRE n.id IS UNIQUE
            """)
            
        driver.close()
        return True
    except Exception as e:
        st.error(f"Neo4j Connection Error: {str(e)}")
        return False

def extract_pdf_text(pdf_docs):
    """Extract text from PDF documents"""
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def init_neo4j_store(chunks, pdf_name):
    """Initialize Neo4j vector store for a specific PDF"""
    try:
        vector_store = Neo4jVector.from_texts(
            texts=chunks,
            embedding=init_embeddings(),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=f"pdf_index_{pdf_name}",
            node_label=f"PDF_{pdf_name}",
            embedding_node_property="embedding",
            text_node_property="text"
        )
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversation_chain(vector_store):
    """Create conversation chain with memory"""
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=init_llm(),
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    
    return conversation_chain

def handle_user_question(user_question):
    """Process user questions and generate responses"""
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    return response['answer']

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", layout="wide")
    
    # Check for HuggingFace token
    if not HUGGINGFACE_TOKEN:
        st.error("HuggingFace token not found. Please add HUGGINGFACE_TOKEN to your .env file.")
        st.stop()
    
    # Add connection status indicator
    if "neo4j_connected" not in st.session_state:
        st.session_state.neo4j_connected = False
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with Multiple PDFs using Neo4j & Groq")
    
    # Test Neo4j connection
    if not st.session_state.neo4j_connected:
        st.session_state.neo4j_connected = test_neo4j_connection()
        if st.session_state.neo4j_connected:
            st.success("Connected to Neo4j successfully!")
        else:
            st.error("Failed to connect to Neo4j. Please check your connection settings.")
            st.stop()
    
    # PDF upload
    pdf_docs = st.file_uploader(
        "Upload your PDFs here",
        type="pdf",
        accept_multiple_files=True
    )
    
    if st.button("Process PDFs"):
        if st.session_state.neo4j_connected:
            with st.spinner("Processing PDFs..."):
                for pdf in pdf_docs:
                    raw_text = extract_pdf_text([pdf])
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = init_neo4j_store(
                        text_chunks,
                        pdf.name.replace('.pdf', '').replace(' ', '_')
                    )
                    
                    if vector_store:
                        st.session_state.conversation = get_conversation_chain(vector_store)
                        st.success(f"Processed {pdf.name} successfully!")
                    else:
                        st.error(f"Failed to process {pdf.name}")
        else:
            st.error("Please ensure Neo4j connection is established first.")
    
    # Chat interface
    if st.session_state.conversation is not None:
        user_question = st.text_input("Ask a question about your PDFs:")
        
        if user_question:
            with st.spinner("Thinking..."):
                response = handle_user_question(user_question)
                
            # Display chat history
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write("Human: ", message.content)
                else:
                    st.write("Assistant: ", message.content)

if __name__ == "__main__":
    main()
