# Import necessary libraries for Streamlit, LangChain, and document processing
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
import os
import tempfile

def initialize_session_state():
    # Initialize Streamlit session state variables to manage chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        # Initial bot greeting
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]
    if 'past' not in st.session_state:
        # Initial user greeting
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    # Process user query using the conversational retrieval chain
    result = chain({"question": query, "chat_history": history})
    # Add current query and response to conversation history
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    # Create containers for chat interface
    reply_container = st.container()
    container = st.container()
    
    with container:
        # Create a form for user input
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')
        
        # Process user input when submitted
        if submit_button and user_input:
            with st.spinner('Generating response...'):
                # Generate response using conversational chain
                output = conversation_chat(user_input, chain, st.session_state['history'])
            
            # Update session state with user input and bot response
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    
    # Display chat history with user and bot messages
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                # Display user messages
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                # Display bot messages
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    # Initialize LLM (Large Language Model) with specific parameters
    llm = LlamaCpp(
        streaming = True,
        model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.75,
        top_p=1, 
        verbose=True,
        n_ctx=4096
    )
    
    # Create conversation memory to maintain context
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    return chain

def main():
    # Initialize session state variables
    initialize_session_state()
    
    # Set up Streamlit app title
    st.title("Free LLM Chatbot")
    
    # Create sidebar for file upload
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    
    # Process uploaded files
    if uploaded_files:
        text = []
        for file in uploaded_files:
            # Get file extension
            file_extension = os.path.splitext(file.name)[1]
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            
            # Load PDF files
            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            
            # Extract text from loaded documents
            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)
        
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(text)
        
        # Create embeddings using HuggingFace model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector store from text chunks
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        
        # Create conversational chain
        chain = create_conversational_chain(vector_store)
        
        # Display chat interface
        display_chat_history(chain)

# Run the main application
if __name__ == "__main__":
    main()
