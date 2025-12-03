# Test code for document ingestion and analysis using a PDFHandler and DocumentAnalyser

# import os
# from pathlib import Path
# from src.document_analyser.data_ingestion import DocumentHandler
# from src.document_analyser.data_analysis import DocumentAnalyser

# # Path to the PDF you want to test
# # PDF_PATH = r"D:\\Agentic_AI\\Practice\\DOCUMENT_PORTAL_1\\data\\document_analysis\\NIPS-2017-attention-is-all-you-need-Paper.pdf"
# PDF_PATH = r"D:\Agentic_AI\Practice\DOCUMENT_PORTAL_1\data\document_analysis\sample.pdf"

# # Dummy file wrapper to simulate uploaded file (streamlit style)


# class DummyFile:
#     def __init__(self, file_path):
#         self.name = Path(file_path).name
#         self._file_path = file_path

#     def getbuffer(self):
#         return open(self._file_path, 'rb').read()


# def main():
#     try:
#         # --------------STEP-1: DATA INGESTION----------------
#         print("Starting Data Ingestion...")
#         dummy_pdf = DummyFile(PDF_PATH)

#         handler = DocumentHandler(session_id="test_ingestion_analysis")
#         saved_path = handler.save_pdf(dummy_pdf)
#         print(f"PDF saved at: {saved_path}")

#         text_content = handler.read_pdf(saved_path)
#         print(f"Extracted Text Length: {len(text_content)} chars\n")

#         # --------------STEP-2: DATA ANALYSIS----------------
#         print("Starting Data Analysis...")
#         analyser = DocumentAnalyser()  # Loads LLM + Parser
#         analysis_result = analyser.analyze_document(text_content)

#         # --------------STEP-3: DISPLAY RESULTS----------------
#         print("\n=== METADATA ANALYSIS RESULT ===")
#         for key, value in analysis_result.items():
#             print(f"{key}: {value}\n")
#     except Exception as e:
#         print(f"Test failed: {e}")


# if __name__ == "__main__":
#     main()

# Test code for document comparison using LLMs

# import io
# from pathlib import Path
# from src.document_compare.data_ingestion import DocumentIngestion
# from src.document_compare.document_comparartor import DocumentComparatorLLM

# # ---- Setup: Load local PDF files as if they were "uploaded" ---- #


# def load_fake_uploaded_file(file_path: str):
#     return io.BytesIO(file_path.read_bytes())  # simulate .getbuffer()

# # ---- Step 1: Save and combine PDFs ---- #


# def test_compare_documents():
#     ref_path = Path(
#         "D:\Agentic_AI\Practice\DOCUMENT_PORTAL_1\data\document_compare\Long_Report_V1.pdf")
#     act_path = Path(
#         "D:\Agentic_AI\Practice\DOCUMENT_PORTAL_1\data\document_compare\Long_Report_V2.pdf")

#     class FakeUpload:
#         def __init__(self, file_path: Path):
#             self.name = file_path.name
#             self._buffer = file_path.read_bytes()

#         def getbuffer(self):
#             return self._buffer

#     comparator = DocumentIngestion()
#     ref_upload = FakeUpload(ref_path)
#     act_upload = FakeUpload(act_path)

#     # Save files and combine
#     ref_file, act_file = comparator .save_uploaded_file(
#         ref_upload, act_upload)
#     combined_text = comparator.combine_documents()
#     comparator.clean_old_sessions(keep_latest=2)

#     print("\n Combined Text Preview (First thousand chars): \n")
#     print(combined_text[:1000])

#     # ---- Step 2: Run LLM comparison ---- #
#     llm_comparator = DocumentComparatorLLM()
#     df = llm_comparator.compare_documents(combined_docs=combined_text)

#     print("\n=== Comparison DataFrame ===")
#     print(df)


# if __name__ == "__main__":
#     test_compare_documents()

# Testing code for document chat functionality

# import sys
# from pathlib import Path
# from langchain_community.vectorstores import FAISS
# from src.single_document_chat.data_ingestion import SingleDocIngestor
# from src.single_document_chat.retrieval import CoversationalRAG
# from utils.model_loader import ModelLoader
# from logger.custom_logger import CustomLogger

# FAISS_INDEX_PATH = Path("faiss_index")


# def test_conversational_rag_on_pdf(pdf_path: str, question: str):
#     try:
#         model_loader = ModelLoader()
#         if FAISS_INDEX_PATH.exists():
#             print("Loading existing FAISS index...")
#             embeddings = model_loader.load_embeddings()
#             vectorstore = FAISS.load_local(folder_path=str(
#                 FAISS_INDEX_PATH), embeddings=embeddings, allow_dangerous_deserialization=True)
#             retriever = vectorstore.as_retriever(
#                 search_type="similarity", kwargs={"k": 5})
#         else:
#             # Step2: Ingest document and create retriever
#             print("FAISS index not found. Ingesting PDF and creating index...")
#             with open(pdf_path, "rb") as f:
#                 uploaded_files = [f]
#                 ingestor = SingleDocIngestor()
#                 retriever = ingestor.ingest_files(uploaded_files)
#         print("Running Conversational RAG")
#         session_id = "test_conversational_rag"
#         rag = CoversationalRAG(retriever=retriever, session_id=session_id)

#         response = rag.invoke(question)
#         print(f"\nQuestion: {question}\nAnswer: {response}")

#     except Exception as e:
#         print(f"Test failed: {str(e)}")
#         sys.exit(1)


# if __name__ == "__main__":
#     # Example PDF path and question
#     pdf_path = r"D:\\Agentic_AI\\Practice\\DOCUMENT_PORTAL_1\\data\\single_document_chat\\NIPS-2017-attention-is-all-you-need-Paper.pdf"
#     # question = "What is the main topic of this document?"
#     question = "What is the significance of the attention mechanisms? can you explain it in simple terms"

#     if not Path(pdf_path).exists():
#         print(f"PDF file does not exist at {pdf_path}")
#         sys.exit(1)

#     test_conversational_rag_on_pdf(pdf_path, question)

# Testing multi document chat functionality

import sys
from pathlib import Path
from src.multi_document_chat.data_ingestion import DocumentIngestor
from src.multi_document_chat.retrieval import ConversationalRAG
import os


def test_document_ingestion_and_rag():
    try:
        # os.chdir(r"D:\Agentic_AI\Practice\DOCUMENT_PORTAL_1")
        print("Current working dir:", Path.cwd())
        test_files = [
            "data\\multi_document_chat\\market_analysis_report.docx",
            "data\\multi_document_chat\\NIPS-2017-attention-is-all-you-need-Paper.pdf",
            "data\\multi_document_chat\\sample.pdf",
            "data\\multi_document_chat\\state_of_the_union.txt"
        ]

        # test_files = [
        #     "D:\\Agentic_AI\\Practice\\DOCUMENT_PORTAL_1\\data\\multi_document_chat\\market_analysis_report.docx",
        #     "D:\\Agentic_AI\\Practice\\DOCUMENT_PORTAL_1\\data\\multi_document_chat\\NIPS-2017-attention-is-all-you-need-Paper.pdf",
        #     "D:\\Agentic_AI\\Practice\\DOCUMENT_PORTAL_1\\data\multi_document_chat\\sample.pdf",
        #     "D:\\Agentic_AI\\Practice\\DOCUMENT_PORTAL_1\\data\multi_document_chat\\state_of_the_union.txt"
        # ]

        uploaded_files = []
        for file_path in test_files:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path, "rb"))
            else:
                print(f"File does not exist: {file_path}")

        print("uploaded_files: ", uploaded_files)

        if not uploaded_files:
            print("No valid files to upload. Exiting test.")
            sys.exit(1)

        ingestor = DocumentIngestor()
        retriever = ingestor.ingest_files(uploaded_files)

        for f in uploaded_files:
            f.close()

        session_id = "test_multi_doc_chat"

        rag = ConversationalRAG(retriever=retriever,
                                session_id="test_multi_doc_chat")
        question = "What is attention is all you need paper about?"

        response = rag.invoke(user_input=question)
        print(f"\n Question: {question}")
        print("Answer:", response)
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    test_document_ingestion_and_rag()
