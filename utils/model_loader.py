import os
import sys
from dotenv import load_dotenv

from utils.config_loader import load_config

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from logger.custom_logger import CustomLogger
from exception.custom_exception_archive import DocumentPortalException
log = CustomLogger().get_logger(__name__)


class ModelLoader:
    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.config = load_config()
        log.info("Configuration loaded successfully",
                 config_keys=list(self.config.keys()))

    def _validate_env(self):
        """
        Validate necessary environment variables.
        Esnure API keys exist.
        Raises DocumentPortalException if any required variable is missing.
        """
        required_vars = ["GOOGLE_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"]
        self.api_keys = {key: os.getenv(key) for key in required_vars}
        missing = [key for key, value in self.api_keys.items() if not value]
        if missing:
            log.error("Missing environment variables", missing_vars=missing)
            raise DocumentPortalException(
                "Missing required environment variables", sys
            )
        log.info("Environment variables validated", available_keys=[
                 k for k in self.api_keys if self.api_keys[k]])

    def load_embeddings(self):
        """
        Load and return the embeddings model.
        """
        try:
            log.info("Loading embeddings model...")
            model_name = self.config["embedding_model"]["model_name"]
            return GoogleGenerativeAIEmbeddings(model=model_name)
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise

    def load_llm(self):
        """
        Load and return llm model.
        """
        """Load LLM dynamically based on provider in config."""
        llm_block = self.config["llm"]
        # Default provider ya ENV var se choose karo
        provider_key = os.getenv("LLM_PROVIDER", "groq")  # Default groq

        if provider_key not in llm_block:
            log.error("LLM provider not found in config",
                      provider_key=provider_key)
            raise ValueError(f"Provider '{provider_key}' not found in config.")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider,
                 model_name=model_name, temperature=temperature, max_tokens=max_tokens)

        if provider == "google":
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            return llm

        elif provider == "groq":
            llm = ChatGroq(
                model=model_name,
                api_key=self.api_keys["GROQ_API_KEY"],
                temperature=temperature,
            )
            return llm

        elif provider == "openai":
            llm = ChatOpenAI(
                model_name=model_name,
                api_key=self.api_keys["OPENAI_API_KEY"],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return llm
        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    loader = ModelLoader()

    # Test embeddings model Loading
    embeddings = loader.load_embeddings()
    print("Embeddings model loaded successfully.")

    # Check the embedding result
    result = embeddings.embed_query("Hello, How are you!")
    print(f"Embeddings result: {result}")

    # Test LLM model Loading
    llm = loader.load_llm()
    print("LLM model loaded successfully.")

    # Test the modelloader
    result = llm.invoke("Hello, how are you?")
    print("LLM result:", result)
