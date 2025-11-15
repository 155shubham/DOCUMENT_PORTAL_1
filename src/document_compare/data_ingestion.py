import sys
from pathlib import Path
import fitz  # PyMuPDF
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from datetime import datetime, timezone
import uuid


class DocumentIngestion:
    """
    Handles saving, reading, and combining of PDFs for comparison with session-based versioning.
    """

    def __init__(self, base_dir: str = "data\\document_compare", session_id=None):
        """
        Initializes the DocumentAnalyser with necessary configurations."""
        self.log = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)
        # self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.log.info("DocumentIngestion initialised",
                      session_path=str(self.session_path))

    def delete_existing_files(self):
        """
        Deletes existing files at the specified paths.
        """
        try:
            if self.session_path.exists() and self.session_path.is_dir():
                for file in self.session_path.iterdir():
                    if file.is_file():
                        file.unlink()
                        self.log.info("File deleted", file=str(file))
                self.log.info("Directory cleaned up",
                              directory=str(self.base_dir))
        except Exception as e:
            self.log.error(f"Error in delete_existing_files: {e}")
            raise DocumentPortalException(
                "An error occurred while deleting existing files.", sys)

    def save_uploaded_file(self, reference_file, actual_file):
        """
        Save reference and actual PDF files in the session directory.
        """
        try:
            ref_path = self.session_path / reference_file.name
            act_path = self.session_path / actual_file.name

            if not reference_file.name.lower().endswith(".pdf") or not actual_file.name.lower().endswith(".pdf"):
                raise ValueError("Only PDF files are allowed.")

            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())

            with open(act_path, "wb") as f:
                f.write(actual_file.getbuffer())

            self.log.info("Files saved",
                          reference=str(ref_path), actual=str(act_path), session=self.session_id)
            return ref_path, act_path

        except Exception as e:
            self.log.error("Error saving PDF files", error=str(e))
            raise DocumentPortalException(
                "Error saving files.", sys)

    def read_pdf(self, pdf_path: Path) -> str:
        """
        Read text content of a PDF page-by-page.
        """
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError("PDF is encrypted: {pdf_path.name}")

                all_text = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        all_text.append(
                            f"\n --- Page {page_num + 1} --- \n{text}")

            self.log.info("PDF read successfully", file=str(
                pdf_path), pages=len(all_text))
            return "\n".join(all_text)

        except Exception as e:
            self.log.error("Error reading pdf",
                           file=str(pdf_path), error=str(e))
            raise DocumentPortalException(
                "Error reading pdf", sys)

    def combine_documents(self) -> str:
        """
        Combine content of all PDFs in session folder into a single string.
        """
        doc_parts = []
        try:
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() == ".pdf":
                    content = self.read_pdf(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")

            combined_text = "\n\n".join(doc_parts)
            self.log.info("Documents combined", count=len(doc_parts))
            return combined_text

        except Exception as e:
            self.log.error("Error combining documents",
                           error=str(e), session=self.session_id)
            raise DocumentPortalException(
                "Error combining documents", sys)

    def clean_old_sessions(self, keep_latest: int = 3):
        """
        Optional method to delete older sessions, keeping only the latest N.
        """
        try:
            session_folders = sorted(
                [d for d in self.base_dir.iterdir() if d.is_dir()],
                reverse=True
            )

            for folder in session_folders[keep_latest:]:
                for file in folder.iterdir():
                    if file.is_file():
                        file.unlink()
                folder.rmdir()
                self.log.info("Old session folder deleted", folder=str(folder))

        except Exception as e:
            self.log.error(f"Error cleaning old sessions: {e}")
            raise DocumentPortalException("Error cleaning old sessions.", sys)
