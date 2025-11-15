import os
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY


class DocumentAnalyser:
    """
    Analyzes documents using pre-trained models.
    Automatically logs all actions and supports session-based organization.
    """

    def __init__(self, data_dir=None, session_id=None):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()

            # Prepare Parsers
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(
                parser=self.parser, llm=self.llm)

            self.prompt = PROMPT_REGISTRY["document_analysis"]

            self.log.info("DocumentAnalyser initialized successfully.")

        except Exception as e:
            self.log.error(f"Error initializing DocumentAnalyser: {e}")
            raise DocumentPortalException(
                "Error in DocumentAnalyser initialization", error_details=str(e)) from e

    def analyze_document(self, document_text: str) -> dict:
        """
        Analyse a document's text and extract stuctured metadata and summary.
        """
        try:
            chain = self.prompt | self.llm | self.fixing_parser

            self.log.info("Metadata analysis chain initialized.")

            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_content": document_text
            })

            self.log.info("Metadata extraction successful.",
                          keys=list(response.keys()))
            return response
        except Exception as e:
            self.log.error("Metadata analysis failed", error=str(e))
            raise DocumentPortalException(
                "Metadata extraction failed", error_details=str(e)) from e


"""
Explannation for Chain invoker:
Execution flow (what happens when chain.invoke(...) runs)
Preconditions
chain was created as self.prompt | self.llm | self.fixing_parser (so each component must support or and the resulting object must expose invoke(payload)).
payload contains:
"format_instructions": self.parser.get_format_instructions()
"document_content": document_text

Step‑by‑step execution
Prompt component receives the payload and formats it into the prompt the LLM expects (inserting format_instructions and document_content). This step may be a simple template formatter or a LangChain PromptTemplate wrapper.
LLM component is invoked with the formatted prompt. The LLM produces output (string, tokens, or model response object).
Fixing parser (OutputFixingParser) receives the raw LLM output and attempts to parse it into the expected structured format (using the JsonOutputParser/Metadata pydantic model). If the raw output is invalid JSON, the fixing parser may call the LLM again or apply heuristics to correct it and produce valid JSON.
The final parsed result (typically a dict matching the Metadata schema) is returned by chain.invoke and assigned to response.

"""
