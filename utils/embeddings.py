import logging
from typing import Optional
from langchain.embeddings.base import Embeddings
from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embeddings(model: str = "text-embedding-3-small") -> Embeddings:
    """
    Initialize and return OpenAI embeddings model.
    
    Args:
        model (str): The OpenAI embedding model to use. 
                    Defaults to 'text-embedding-3-small'.
    
    Returns:
        Embeddings: An instance of OpenAI embeddings
    
    Raises:
        ImportError: If required packages are not installed
        Exception: For other initialization errors
    """
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        logger.error("Required package 'langchain-openai' not found")
        raise ImportError(
            "Could not import langchain_openai. "
            "Please install it with: pip install langchain-openai"
        )

    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=Config.OPENAI_API_KEY,
            model=model,
            chunk_size=1000,  # Process 1000 texts at a time
            max_retries=3,    # Retry failed requests up to 3 times
        )
        logger.info(f"Successfully initialized OpenAI embeddings with model: {model}")
        return embeddings

    except ValueError as ve:
        logger.error(f"Invalid configuration: {str(ve)}")
        raise

    except Exception as e:
        logger.error(f"Error initializing OpenAI embeddings: {str(e)}")
        raise Exception(
            f"Failed to initialize OpenAI embeddings: {str(e)}. "
            "Please check your API key and internet connection."
        )

def get_alternative_embeddings() -> Optional[Embeddings]:
    """
    Fallback function to provide alternative embedding models if OpenAI is unavailable.
    Currently supports HuggingFace sentence-transformers as a fallback.
    
    Returns:
        Optional[Embeddings]: An alternative embedding model instance or None if unavailable
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info(f"Successfully initialized HuggingFace embeddings with model: {model_name}")
        return embeddings

    except ImportError:
        logger.warning("HuggingFace sentence-transformers not available for fallback")
        return None

    except Exception as e:
        logger.error(f"Error initializing alternative embeddings: {str(e)}")
        return None