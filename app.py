import os
import streamlit as st
from pdf_processing import load_and_process_pdf
from vector_store import VectorStoreManager
from chatbot import get_qa_chain
import torch
import tempfile
import shutil
import dateutil.parser

torch.classes.__path__ = []


def initialize_session_state():
    """Initialize all session state variables"""
    if "selected_pdf" not in st.session_state:
        st.session_state.selected_pdf = None
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = VectorStoreManager()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_file_info" not in st.session_state:
        st.session_state.current_file_info = None


def display_file_info():
    """Display information about the currently loaded file"""
    if st.session_state.current_file_info:
        with st.sidebar.expander("📊 Document Info", expanded=True):
            info = st.session_state.current_file_info
            st.write(f"Chunks: {info.get('num_chunks', 'N/A')}")
            last_processed = info.get('last_processed')
            if last_processed:
                try:
                    # Menggunakan dateutil.parser untuk kompatibilitas yang lebih baik
                    dt = dateutil.parser.parse(last_processed)
                    st.write(f"Last processed: {dt.strftime('%Y-%m-%d %H:%M')}")
                except Exception:
                    st.write(f"Last processed: {last_processed}")


def main():
    st.title("📄 Chat with Your PDF Transcript/Text")
    st.sidebar.header("📂 Upload or Select an Existing PDF")

    # Initialize session state
    initialize_session_state()

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # File uploader
        uploaded_file = st.sidebar.file_uploader("🔍 Upload PDF:", type=["pdf"])

        if uploaded_file is not None:
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File too large! Please upload a file less than 10MB.")
            else:
                temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.read())

                st.session_state.selected_pdf = temp_pdf_path

        else:
            # Select existing PDF
            existing_pdfs = [f for f in os.listdir() if f.endswith(".pdf")]
            selected_pdf = st.sidebar.selectbox(
                "Select an existing PDF:",
                existing_pdfs
            ) if existing_pdfs else None

            if selected_pdf:
                st.session_state.selected_pdf = selected_pdf

        # Process PDF if selected
        if st.session_state.selected_pdf:
            with st.sidebar:
                with st.spinner("⏳ Processing document..."):
                    # Load and process PDF
                    docs = load_and_process_pdf(st.session_state.selected_pdf)

                    # Process documents and update vector store
                    was_processed = st.session_state.vector_store_manager.process_document(
                        st.session_state.selected_pdf,
                        docs
                    )

                    # Update file info
                    st.session_state.current_file_info = (
                        st.session_state.vector_store_manager.get_file_info(
                            st.session_state.selected_pdf
                        )
                    )

                    if was_processed:
                        st.success("🔍 Document processed and indexed!")
                    else:
                        st.success("🔍 Using existing document index!")

                # Display file info
                display_file_info()

                # Clear chat history button
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()

                # Clear all data button
                if st.button("🧹 Clear All Data"):
                    st.session_state.vector_store_manager.clear_all()
                    st.session_state.chat_history = []
                    st.session_state.current_file_info = None
                    st.rerun()

        # Chatbot Interface
        if st.session_state.selected_pdf:
            vector_store = st.session_state.vector_store_manager.get_vector_store()
            qa_chain = get_qa_chain(vector_store)

            # question_templates = [
            #     "What is the summary of the document?",
            #     "Can you explain the main points in the document?",
            #     "What are the key takeaways from the document?",
            #     "Please explain the first section of the document.",
            #     "What does the document say about [insert topic]?"
            # ]
            #
            # selected_template = st.sidebar.selectbox(
            #     "Choose a question template",
            #     ["Select a question template"] + question_templates
            # )

            question_templates = [
                "What is the summary of the document?",
                "Can you explain the main points in the document?",
                "What are the key takeaways from the document?",
                "Please explain the first section of the document.",
                "What does the document say about [insert topic]?"
            ]

            st.markdown("<h3>Choose a question from the templates below:</h3>", unsafe_allow_html=True)

            selected_template = None
            for template in question_templates:
                if st.button(template):
                    selected_template = template

            user_query = st.chat_input("💬 Ask a question about the document:")

            # if selected_template != "Select a question template":
            #     user_query = selected_template

            if selected_template:
                user_query = selected_template

            if user_query:
                with st.spinner("🤔 Thinking..."):
                    try:
                        response = qa_chain.invoke(user_query)

                        # Extract answer and source documents
                        answer = response['result'] if isinstance(response, dict) else response
                        sources = response.get('source_documents', []) if isinstance(response, dict) else []

                        # Add to chat history with emojis
                        st.session_state.chat_history.append(("🧑‍💻", user_query))  # User emoji
                        st.session_state.chat_history.append(("🤖", answer))  # Bot emoji

                        # Display source information if available
                        if sources:
                            with st.expander("📚 Sources"):
                                for idx, doc in enumerate(sources, 1):
                                    st.write(f"Source {idx}:")
                                    st.write(doc.page_content[:200] + "...")

                    except Exception as e:
                        st.error(f"Error processing your question: {str(e)}")

            # Display chat history
            for role, text in st.session_state.chat_history:
                st.write(f"**{role}:** {text}")

    finally:
        # Cleanup temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
