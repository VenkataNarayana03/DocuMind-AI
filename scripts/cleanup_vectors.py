import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
load_dotenv(BACKEND / ".env")

from app.core.settings import get_settings  # noqa: E402
from app.services.rag.pipeline import pipeline  # noqa: E402


settings = get_settings()
pipeline.cleanup_session()

if settings.pinecone_api_key:
    from pinecone import Pinecone

    pc = Pinecone(api_key=settings.pinecone_api_key)
    pc.Index(settings.pinecone_index).delete(delete_all=True)
    print(f"Cleared all vectors from Pinecone index '{settings.pinecone_index}'.")
else:
    print("Pinecone is not configured; cleared only current local session state.")
