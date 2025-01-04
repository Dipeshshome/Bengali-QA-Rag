import logging
from pathlib import Path
from utils.document_loader import load_document
from utils.text_splitter import split_documents
from utils.embeddings import get_embeddings
from database.vector_db import VectorDatabase
from models.llm import get_llm
from rag.qa_chain import BengaliQAChain

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # File path
        file_path = "FAQ.pdf"
        
        # Check if file exists and is PDF
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if Path(file_path).suffix.lower() != '.pdf':
            raise ValueError("Only PDF files are currently supported")
        
        # Load document
        logger.info("Loading Bengali document...")
        documents = load_document(file_path)
        
        # Split into chunks
        logger.info("Splitting documents into chunks...")
        chunks = split_documents(documents)
        
        # Initialize embeddings with Bengali configuration
        logger.info("Initializing embeddings...")
        embedding_function = get_embeddings(
            model="text-embedding-3-small"  # Latest model with better multilingual support
        )
        
        # Initialize and create vector database
        logger.info("Creating vector database...")
        db = VectorDatabase(embedding_function)
        
        # Try to load existing DB, create new one if doesn't exist
        vector_store = db.load_db()
        if vector_store is None:
            vector_store = db.create_db(chunks)
        
        # Initialize LLM with Bengali configuration
        logger.info("Initializing LLM...")
        llm = get_llm(model_name="gpt-4-turbo-preview")  # Using GPT-4 for better Bengali understanding
        
        # Create Bengali QA chain
        logger.info("Creating Bengali QA chain...")
        qa_chain = BengaliQAChain(llm, vector_store.as_retriever())
        
        print("\nবাংলা প্রশ্ন-উত্তর সিস্টেমে স্বাগতম!")  # Welcome to Bengali Q&A system!
        print("প্রশ্ন করতে টাইপ করুন, বের হতে 'quit' লিখুন।")  # Type your question, write 'quit' to exit.
        
        while True:
            # Get question from user
            question = input("\nআপনার প্রশ্ন লিখুন: ")  # Write your question:
            
            if question.lower() == 'quit':
                print("\nধন্যবাদ! আবার আসবেন।")  # Thank you! Come again.
                break
                
            logger.info(f"Processing Bengali question: {question}")
            answer = qa_chain.get_answer(question)
            print("\nউত্তর:", answer)  # Answer:

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print("\nদুঃখিত, একটি সমস্যা হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।")  # Sorry, there was a problem. Please try again.
        raise

if __name__ == "__main__":
    main()