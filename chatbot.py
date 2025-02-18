import os
from langchain_together import ChatTogether
from langchain.chains import RetrievalQA
from vector_store import VectorStoreManager
from config import LLM_MODEL, TOGETHER_API_KEY
import logging
from typing import Optional
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3

class ChatbotInitError(Exception):
    """Custom exception for chatbot initialization errors."""
    pass


def validate_api_key() -> None:
    """Validate that the Together API key is available."""
    if not TOGETHER_API_KEY:
        raise ChatbotInitError("Together API key not found. Please set TOGETHER_API_KEY in your environment variables.")


def get_qa_chain(vector_store: Chroma) -> RetrievalQA:
    """
    Initialize a chatbot with the Llama-2 model and provided vector store.

    Args:
        vector_store: Initialized Chroma vector store

    Returns:
        RetrievalQA: Initialized QA chain

    Raises:
        ChatbotInitError: If initialization fails
    """
    try:
        # Validate API key
        validate_api_key()

        # Configure retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3,  # Retrieve top 3 most relevant chunks
                # "include_metadata": True
            }
        )

        # Initialize LLM
        llm = ChatTogether(
            model=LLM_MODEL,
            temperature=0.3,
            together_api_key=TOGETHER_API_KEY,
            max_tokens=2000,
            top_p=0.9,
            # context_window=4096
        )

        # Create prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create QA chain with custom prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": PROMPT
            },
            return_source_documents=True,
            verbose=True
        )

        logging.info("✅ Successfully initialized QA chain")
        return qa_chain

    except Exception as e:
        error_msg = f"Failed to initialize QA chain: {str(e)}"
        logging.error(f"❌ {error_msg}")
        raise ChatbotInitError(error_msg)
