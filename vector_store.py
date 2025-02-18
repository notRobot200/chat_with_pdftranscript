from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DB_DIR, EMBEDDING_MODEL
import logging
import hashlib
import json
import os
from typing import List, Dict
from langchain.docstore.document import Document
import datetime


class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings
        )
        self.cache_file = os.path.join(CHROMA_DB_DIR, "processed_files.json")
        self._load_cache()

    def _load_cache(self):
        """Load cache of processed files"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.processed_files = json.load(f)
        else:
            self.processed_files = {}
            self._save_cache()

    def _save_cache(self):
        """Save cache of processed files"""
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.processed_files, f)

    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file content"""
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()

    def _remove_existing_documents(self, file_path: str):
        """Remove existing documents for a given file from the vector store"""
        try:
            # Get all documents
            result = self.vector_db.get()
            if not result or 'ids' not in result or 'metadatas' not in result:
                return

            # Find IDs of documents from this file
            ids_to_delete = [
                id for id, metadata in zip(result['ids'], result['metadatas'])
                if metadata.get('source_file') == file_path
            ]

            if ids_to_delete:
                logging.info(f"🗑️ Removing {len(ids_to_delete)} existing documents for {file_path}")
                self.vector_db.delete(ids_to_delete)

        except Exception as e:
            logging.error(f"❌ Error removing existing documents: {str(e)}")
            raise

    def process_document(self, file_path: str, docs: List[Document]) -> bool:
        """
        Process document and update vector store.
        Returns True if document was processed, False if it was loaded from cache.
        """
        try:
            current_hash = self._get_file_hash(file_path)
            file_info = self.processed_files.get(file_path, {})

            # Check if file has been processed and hasn't changed
            if file_info.get('hash') == current_hash:
                logging.info(f"📂 Document {file_path} hasn't changed, using existing vectors")
                return False

            # Remove existing documents for this file
            self._remove_existing_documents(file_path)

            # Add new documents with metadata
            for doc in docs:
                doc.metadata['file_hash'] = current_hash
                doc.metadata['source_file'] = file_path

            # Store new documents
            self.vector_db.add_documents(docs)

            # Update cache
            self.processed_files[file_path] = {
                'hash': current_hash,
                'num_chunks': len(docs),
                'last_processed': datetime.datetime.now().isoformat()
            }
            self._save_cache()

            logging.info(f"✅ Added {len(docs)} new chunks from {file_path} to vector store")
            return True

        except Exception as e:
            logging.error(f"❌ Error processing document {file_path}: {str(e)}")
            raise

    def get_vector_store(self) -> Chroma:
        """Get the vector store instance"""
        return self.vector_db

    def get_file_info(self, file_path: str) -> Dict:
        """Get information about processed file"""
        return self.processed_files.get(file_path, {})

    def clear_all(self):
        """Clear all documents from vector store and cache"""
        try:
            # Clear vector store
            self.vector_db.delete(self.vector_db.get()['ids'])

            # Clear cache
            self.processed_files = {}
            self._save_cache()

            logging.info("🧹 Cleared all documents from vector store and cache")
        except Exception as e:
            logging.error(f"❌ Error clearing vector store: {str(e)}")
            raise