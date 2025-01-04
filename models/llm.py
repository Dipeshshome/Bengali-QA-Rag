import logging
from typing import Optional
from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm(model_name: str = None):
    """
    Initialize and return OpenAI ChatGPT model.
    
    Args:
        model_name (str, optional): The specific model to use. 
                                  Defaults to value in Config.MODEL_NAME
    
    Returns:
        ChatOpenAI: An instance of OpenAI chat model
    
    Raises:
        ImportError: If required packages are not installed
        Exception: For other initialization errors
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        logger.error("Required package 'langchain-openai' not found")
        raise ImportError(
            "Could not import langchain_openai. "
            "Please install it with: pip install langchain-openai openai"
        )

    try:
        model = model_name or Config.MODEL_NAME
        llm = ChatOpenAI(
            model_name=model,
            openai_api_key=Config.OPENAI_API_KEY,
            temperature=0.7,
            max_retries=3,
        )
        logger.info(f"Successfully initialized ChatOpenAI with model: {model}")
        return llm

    except ValueError as ve:
        logger.error(f"Invalid configuration: {str(ve)}")
        raise

    except Exception as e:
        logger.error(f"Error initializing ChatOpenAI: {str(e)}")
        raise Exception(
            f"Failed to initialize ChatOpenAI: {str(e)}. "
            "Please check your API key and internet connection."
        )