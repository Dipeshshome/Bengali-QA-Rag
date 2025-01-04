from langchain_community.vectorstores import Chroma
from typing import List
from langchain.schema import Document
from config.config import Config

class ChromaDatabase:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.db_path = Config.CHROMA_DB_DIR

    def create_db(self, documents: List[Document]):
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.db_path
        )

    def load_db(self):
        return Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_function
        )