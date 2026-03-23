from fastapi import APIRouter, Query
from model_registry import get_all_models

router = APIRouter(prefix="/models", tags=["Models"])

@router.get("/")
def list_models(source: str = Query(default=None, description="google | huggingface")):
    """Returns all registered models. Used by the UI to build the dropdown."""
    return get_all_models(source=source)
