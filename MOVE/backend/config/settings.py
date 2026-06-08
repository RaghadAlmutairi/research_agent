import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Crawling
    firecrawl_api_key: str = os.getenv("FIRECRAWL_API_KEY", "")
    crawl_requests_per_second: float = float(os.getenv("CRAWL_RPS", "2"))
    crawl_concurrency: int = int(os.getenv("CRAWL_CONCURRENCY", "5"))
    crawl_limit_per_site: int = int(os.getenv("CRAWL_LIMIT", "100"))

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # ChromaDB
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "knowledge_base")

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./move.db")

    # LangSmith (optional)
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
    langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "MOVE-Research-Agent")


settings = Settings()
