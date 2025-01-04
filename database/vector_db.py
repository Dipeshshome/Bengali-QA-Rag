import logging
from typing import List
from langchain.schema import Document
from langchain.vectorstores import FAISS
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.db_path = "faiss_index"

    def create_db(self, documents: List[Document]):
        """
        Create a new FAISS vector store from documents
        """
        try:
            logger.info("Creating new vector store...")
            db = FAISS.from_documents(documents, self.embedding_function)
            
            # Save the vector store
            self.save_db(db)
            logger.info("Vector store created and saved successfully")
            
            return db
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    def load_db(self):
        """
        Load existing vector store if it exists
        """
        try:
            if os.path.exists(f"{self.db_path}.faiss"):
                logger.info("Loading existing vector store...")
                return FAISS.load_local(
                    self.db_path,
                    self.embedding_function
                )
            else:
                logger.warning("No existing vector store found")
                return None
                
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    def save_db(self, db):
        """
        Save the vector store to disk
        """
        try:
            logger.info(f"Saving vector store to {self.db_path}")
            db.save_local(self.db_path)
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise