"""
adapter_router.py
-----------------
Maps model source → correct adapter.
Adding audio later = add audio adapter here.
"""

from .google_adapter import GoogleAdapter
from .huggingface_adapter import HuggingFaceAdapter
from .base_adapter import BaseTextAdapter

# One instance per source — reused across all requests
_registry: dict[str, BaseTextAdapter] = {
    "google":       GoogleAdapter(),
    "huggingface":  HuggingFaceAdapter(),
    # "audio_google": AudioGoogleAdapter(),   ← add later
}

def get_adapter(source: str) -> BaseTextAdapter:
    adapter = _registry.get(source)
    if not adapter:
        raise ValueError(f"No adapter for source '{source}'. Available: {list(_registry.keys())}")
    return adapter
