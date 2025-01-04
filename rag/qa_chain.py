from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BengaliQAChain:
    def __init__(self, llm, retriever):
        # Bengali-specific prompt template
        self.prompt_template = """You are a helpful assistant that answers questions in Bengali language.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say "আমি দুঃখিত, আমি এই প্রশ্নের উত্তর জানি না।" (I'm sorry, I don't know the answer to this question.)
        Always answer in Bengali language.
        
        संदर्भ (Context):
        {context}
        
        প্রশ্ন (Question):
        {question}
        
        উত্তর (Answer) in Bengali:"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": self.prompt,
                "verbose": True
            }
        )
    
    def get_answer(self, question: str) -> str:
        """
        Get answer for a Bengali question.
        
        Args:
            question (str): Question in Bengali
            
        Returns:
            str: Answer in Bengali
        """
        try:
            logger.info(f"Processing Bengali question: {question}")
            response = self.chain.run(question)
            
            # Ensure response is in Bengali
            if not any('\u0980' <= c <= '\u09FF' for c in response):
                logger.warning("Response might not be in Bengali, applying translation prompt")
                # Add translation prompt if response is not in Bengali
                translation_prompt = f"Please translate this answer to Bengali: {response}"
                response = self.chain.llm.predict(translation_prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return "দুঃখিত, একটি ত্রুটি ঘটেছে। অনুগ্রহ করে আবার চেষ্টা করুন।"