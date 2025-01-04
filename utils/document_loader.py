import logging
from typing import List
from pathlib import Path
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_document(file_path: str) -> List[Document]:
    """
    Load PDF documents using PyPDFLoader.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        List[Document]: List of loaded documents
        
    Raises:
        Exception: For loading errors
    """
    try:
        # Convert file path to Path object
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() != '.pdf':
            raise ValueError("Only PDF files are supported")

        logger.info(f"Loading document: {file_path}")
        
        # Initialize loader
        loader = PyPDFLoader(str(file_path))
        
        # Load the document
        documents = loader.load()
        
        logger.info(f"Successfully loaded {len(documents)} pages")
        return documents

    except Exception as e:
        logger.error(f"Error loading document: {str(e)}")
        raise Exception(f"Failed to load document: {str(e)}")