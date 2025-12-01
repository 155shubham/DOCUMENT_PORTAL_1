import uuid
from pathlib import Path
import sys
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader
from datetime import datetime, timezone

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class DocumentIngestor:
    def __init__(self, temp_dir: str = "data/multi_document_chat", faiss_dir: str = "faiss_index", sesion_id: str | None = None):
        try:
            self.SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS
            self.log = CustomLogger().get_logger(__name__)
            self.temp_dir = temp_dir
            self.faiss_dir = faiss_dir
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)

            # sessionized paths
            self.sesion_id = sesion_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.sesion_temp_dir = self.temp_dir/self.sesion_id
            self.session_faiss_dir = self.faiss_dir/self.sesion_id
            self.sesion_temp_dir.mkdir(parents=True, exist_ok=True)
            self.session_faiss_dir.mkdir(parents=True, exist_ok=True)

            self.model_loader = ModelLoader()
            self.log.info(
                "DocumentIngestor Initialised",
                temp_base=str(self.temp_dir),
                faiss_base=str(self.faiss_dir),
                session_id=self.sesion_id,
                temp_path=str(self.sesion_temp_dir),
                faiss_path=str(self.session_faiss_dir)
            )

        except Exception as e:
            self.log.error(
                "Failed to initialise DocumentIngestor", error=str(e))
            raise DocumentPortalException(
                "Initialization error in DocumentIngestor", sys)

    def ingest_files(self, uploaded_files):
        # Logic to ingest documents from the specified source
        try:
            documents = []
            for uploaded_file in uploaded_files:
                ext = Path(uploaded_file.name).suffix.lower()
                if ext not in self.SUPPORTED_EXTENSIONS:
                    self.log.warning("Unsupported file skipped",
                                     filename=uploaded_file.name)
                    continue
                temp_path = self.sesion_temp_dir/uploaded_file.name

                with open(temp_path, "wb") as f_out:
                    f_out.write(uploaded_file.read())
                self.log.info("File saved for ingestion",
                              filename=uploaded_file.name)

                if ext == ".pdf":
                    loader = PyPDFLoader(str(temp_path))
                elif ext == ".docx":
                    loader = Docx2txtLoader(str(temp_path))
                elif ext == ".txt":
                    loader = TextLoader(str(temp_path))
                else:
                    self.log.warning("Unsupported file type encountered",
                                     filename=uploaded_file.name, extension=ext)
                    continue

                docs = loader.load()
                documents.extend(docs)

                if not documents:
                    raise DocumentPortalException(
                        "No valid documents loaded", sys)

                self.log.info(
                    "All documents loaded", totaldocs=len(documents), session_id=self.sesion_id)
            return self._create_retriever(documents)

        except Exception as e:
            self.log.error("Document ingestion failed", error=str(e))
            raise DocumentPortalException(
                "Error during document ingestion", sys)

    def _create_retriever(self, documents):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=300)
            chunks = splitter.split_documents(documents)
            self.log.info("Documents split into chunks", count=len(chunks))

            embeddings = self.model_loader.load_embeddings()
            vectorstore = FAISS.from_documents(
                documents=chunks, embedding=embeddings)
            self.log.info("FAISS index saved to disk", path=str(
                self.session_faiss_dir), session_id=self.sesion_id)

            # save FAISS Index under session folder
            vectorstore.save_local(str(self.session_faiss_dir))

            retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 5})
            self.log.info("Retriever created successfully",
                          retriever_type=str(type(retriever)))

            return retriever
        except Exception as e:
            self.log.error("Failed to create retriever", error=str(e))
            raise DocumentPortalException(
                "Error during retriever creation", sys)
