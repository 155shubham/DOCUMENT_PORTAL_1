import os
import sys
from pathlib import Path
from langchain_community.vectorstores import FAISS
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from utils.model_loader import ModelLoader
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType
from langchain.prompts.chat import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from typing import List, Optional
from langchain_core.messages import BaseMessage


class ConversationalRAG:
    def __init__(self, session_id: str | None = None, retriever=None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA]

            if retriever is None:
                raise ValueError("Retriever cannot be None")
            self.retriever = retriever
            self._build_lcel_chain()
            self.log.info("ConversationalRAG Initialised",
                          session_id=self.session_id)

        except Exception as e:
            self.log.error("Failed to initialize ConversationalRAG",
                           error=str(e))
            raise DocumentPortalException(
                "Initializing Error in ConversationalRAG", sys)

    def load_retriever_from_faiss(self, index_path: str):
        # Logic to load a FAISS retriever from the specified index path
        try:
            embeddings = self.model_loader.load_embeddings()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(
                    f"FAISS index directory not found: {index_path}")

            vectorstore = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True)  # allow_dangerous_deserialization= True  only if you trust the index source)

            self.retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 5})
            self.log.info("FAISS retriever loaded successfully",
                          index_path=index_path, session_id=self.session_id)
            return self.retriever

        except Exception as e:
            self.log.error("Failed to load FAISS retriever", error=str(e))
            raise DocumentPortalException(
                "Error loading FAISS retriever", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """
        Invoke the Conversational RAG chain with the given input and chat history.
        """
        try:
            chat_history = chat_history | []
            payload = {
                "input": input,
                "chat_history": chat_history
            }
            answer = self.chain.invoke(payload)
            if not answer:
                self.log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id)
                return "No answer generated."

            self.log.info("Answer generated successfully",
                          user_input=user_input, session_id=self.session_id)
            return answer
        except Exception as e:
            self.log.error("Failed to invoke ConversationalRAG",
                           error=str(e))
            raise DocumentPortalException(
                "Invocation Error in ConversationalRAG", sys)

    def _load_llm(self):
        # Logic to load the language model
        try:
            self.llm = self.model_loader.load_llm()
            return self.llm
        except Exception as e:
            self.log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("Error loading LLM", sys)

    @staticmethod
    def format_docs(docs):
        # Logic to format documents for retrieval
        try:
            return "\n\n".join(d.page_content for d in docs)
        except Exception as e:
            raise DocumentPortalException("Error formatting documents", sys)

    def _build_lcel_chain(self):
        # Logic to build the LCE (Language-Conditioned Extraction) chain
        try:
            # Rewrite question using chat history
            question_rewriter = {
                {
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history")
                }
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            }

            # Retrieve documents based on rewritten question
            retrieve_docs = question_rewriter | self.retriever | self.format_docs

            # Feed context + original input + chat history into answer prompt
            self.chain = {
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history")
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            }

            self.log.info("LCEL graph built successfully",
                          session_id=self.session_id)

        except Exception as e:
            self.log.error("Failed to build LCE chain", error=str(e))
            raise DocumentPortalException(
                "Chain building error in ConversationalRAG", sys)
