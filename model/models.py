from pydantic import BaseModel, RootModel
from typing import List, Union
from enum import Enum


class Metadata(BaseModel):
    Summary: List[str]
    Title: str
    Author: List[str]
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PagCount: Union[int, str]  # Can be "Not Available"
    SentimentTone: str


class ChangesFormat(BaseModel):
    Pages: str
    Changes: str


class SummaryResponse(RootModel[list[ChangesFormat]]):
    pass


class PromptType(str, Enum):
    DOCUMENT_ANALYSIS = "document_analysis",
    DOCMENT_COMPARISON = "document_compare",
    CONTEXTUALIZE_QUESTION = "contextualize_question",
    CONTEXT_QA = "context_qa"
