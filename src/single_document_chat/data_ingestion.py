import uuid
from pathlib import Path
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader


class SingleDocIngestor:
    def __init__(self, data_dir: str = "data/single_document_chat", faiss_dir: str = "faiss_index"):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.base_dir = Path(data_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.faiss_base_dir = Path(data_dir)
            self.faiss_base_dir.mkdir(parents=True, exist_ok=True)
            self.model_loader = ModelLoader()
            self.log("SingleDocumentIngestor Initialised", temp_path=str(
                self.data_dir), faiss_path=str(faiss_dir))
        except Exception as e:
            self.log.error(
                "Failed to initialise SingleDocIngestor", error=str(e))
            raise DocumentPortalException(
                "Initialization error in SinglDocIngestor", sys)

    def ingest_files(self):
        try:
            pass
        except Exception as e:
            self.log.error("Document ingestion failed", error=str(e))
            raise DocumentPortalException("Error during file ingestion", sys)

    def _create_retriever(self):
        try:
            pass
        except Exception as e:
            self.log.error("Retriever creation failed", error=str(e))
            raise DocumentPortalException(
                "Error creating FAISS retriever", sys)
