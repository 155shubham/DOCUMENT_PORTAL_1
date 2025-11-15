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

import io
from pathlib import Path
from src.document_compare.data_ingestion import DocumentIngestion
from src.document_compare.document_comparartor import DocumentComparatorLLM

# ---- Setup: Load local PDF files as if they were "uploaded" ---- #


def load_fake_uploaded_file(file_path: str):
    return io.BytesIO(file_path.read_bytes())  # simulate .getbuffer()

# ---- Step 1: Save and combine PDFs ---- #


def test_compare_documents():
    ref_path = Path(
        "D:\Agentic_AI\Practice\DOCUMENT_PORTAL_1\data\document_compare\Long_Report_V1.pdf")
    act_path = Path(
        "D:\Agentic_AI\Practice\DOCUMENT_PORTAL_1\data\document_compare\Long_Report_V2.pdf")

    class FakeUpload:
        def __init__(self, file_path: Path):
            self.name = file_path.name
            self._buffer = file_path.read_bytes()

        def getbuffer(self):
            return self._buffer

    comparator = DocumentIngestion()
    ref_upload = FakeUpload(ref_path)
    act_upload = FakeUpload(act_path)

    # Save files and combine
    ref_file, act_file = comparator .save_uploaded_file(
        ref_upload, act_upload)
    combined_text = comparator.combine_documents()
    comparator.clean_old_sessions(keep_latest=2)

    print("\n Combined Text Preview (First thousand chars): \n")
    print(combined_text[:1000])

    # ---- Step 2: Run LLM comparison ---- #
    llm_comparator = DocumentComparatorLLM()
    df = llm_comparator.compare_documents(combined_docs=combined_text)

    print("\n=== Comparison DataFrame ===")
    print(df)


if __name__ == "__main__":
    test_compare_documents()
