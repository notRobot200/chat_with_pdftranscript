import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional
from langchain.docstore.document import Document
import magic
import os


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


def validate_pdf(pdf_path: str) -> None:
    """
    Validate that the file is actually a PDF.

    Args:
        pdf_path: Path to the PDF file

    Raises:
        PDFProcessingError: If file is not a valid PDF
    """
    if not os.path.exists(pdf_path):
        raise PDFProcessingError(f"File not found: {pdf_path}")

    mime = magic.Magic(mime=True)
    file_type = mime.from_file(pdf_path)

    if file_type != 'application/pdf':
        raise PDFProcessingError(f"Invalid file type: {file_type}. Expected PDF file.")


def load_and_process_pdf(
        pdf_path: str,
        chunk_size: int = 1500,
        chunk_overlap: int = 150
) -> List[Document]:
    """
    Load a PDF file and split it into smaller text chunks.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum size of text chunks
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of Document objects containing the processed text chunks

    Raises:
        PDFProcessingError: If PDF processing fails
    """
    try:
        # Validate PDF file
        validate_pdf(pdf_path)

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            raise PDFProcessingError(f"No content found in PDF: {pdf_path}")

        logging.info(f"📂 Loaded {len(documents)} pages from {pdf_path}")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )

        docs = text_splitter.split_documents(documents)

        if not docs:
            raise PDFProcessingError("Failed to create text chunks from PDF")

        # Add source metadata
        for doc in docs:
            doc.metadata['source_file'] = os.path.basename(pdf_path)

        logging.info(f"✅ Created {len(docs)} chunks from {pdf_path}")
        return docs

    except Exception as e:
        error_msg = f"Failed to process PDF {pdf_path}: {str(e)}"
        logging.error(f"❌ {error_msg}")
        raise PDFProcessingError(error_msg)
