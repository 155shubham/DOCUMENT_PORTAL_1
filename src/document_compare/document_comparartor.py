import sys
from dotenv import load_dotenv
import pandas as pd
from logger.custom_logger import CustomLogger
from exception.custom_exception_archive import DocumentPortalException
from model.models import SummaryResponse, PromptType
from prompt.prompt_library import PROMPT_REGISTRY
from utils.model_loader import ModelLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser


class DocumentComparatorLLM:
    """
    Compares two documents using LLMs and pre-trained models.
    Automatically logs all actions and supports session-based organization.
    """

    def __init__(self):
        """
        Initializes the DocumentComparatorLLM with necessary configurations.
        """
        load_dotenv()
        self.log = CustomLogger().get_logger(__name__)
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm()
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(
            parser=self.parser, llm=self.llm)
        self.prompt = PROMPT_REGISTRY[PromptType.DOCMENT_COMPARISON.value]
        self.chain = self.prompt | self.llm | self.parser
        self.log.info(
            "DocumentComparatorLLM initialized.", model=self.llm)

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        """
        Compares two documents and returns a structured comparison.
        """
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instructions": self.parser.get_format_instructions()
            }
            self.log.info("Invoking document comparison LLM chain")
            response = self.chain.invoke(inputs)
            self.log.info("Chain invoked successfully",
                          response_preview=str(response)[:200])
            return self._format_response(response)

        except Exception as e:
            self.log.error("Error in compare_documents", error=str(e))
            raise DocumentPortalException(
                "Error comparing documents", sys)

    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame:
        """
        Formats the LLM response into a structured format.
        """
        try:
            df = pd.DataFrame(response_parsed)
            return df
        except Exception as e:
            self.log.error(
                "Error formatting response into DataFrame", error=str(e))
            raise DocumentPortalException("Error formatting response", sys)
