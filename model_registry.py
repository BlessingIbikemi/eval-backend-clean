"""
model_registry.py
-----------------
THE ONLY FILE YOU NEED TO EDIT to add/remove models.

HOW TO ADD A MODEL:
  - Google base model     → add to MODELS with source="google", type="base"
  - Google fine-tuned     → deploy on Vertex AI first, get endpoint ID, add with source="google", type="finetuned"
  - HuggingFace base      → add with source="huggingface", type="base" (uses HF Inference API, no GPU needed)
  - HuggingFace finetuned → deploy on HF Inference Endpoints first, get URL, add with source="huggingface", type="finetuned"
"""

MODELS = [

    # ──────────────────────────────────────────────────────────────────────
    # GOOGLE MODELS (run on Vertex AI — no GPU needed on your side)
    # ──────────────────────────────────────────────────────────────────────

    {
        "id": "gemini-2.5-pro",
        "label": "Gemini 2.5 Pro",
        "source": "google",
        "type": "base",
        "modality": "text",
        "description": "Google's most capable model",
    },
    {
        "id": "gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
        "source": "google",
        "type": "base",
        "modality": "text",
        "description": "Fast and efficient Gemini 2.5",
    },

    # YOUR FINE-TUNED GOOGLE MODEL
    # Step 1: Go to Vertex AI → Model Registry → Deploy your model to an endpoint
    # Step 2: Copy the endpoint ID (looks like: 1234567890123456789)
    # Step 3: Uncomment and fill in below
    {
        "id": "4363812666817380352",          # just the number ID
        "label": "Fine-tuned Gemini 2.5 Flash",
         "source": "google",
         "type": "finetuned",
         "modality": "text",
         "description": "Fine-tuned on our domain data",
         "vertex_project": "yorubanlp",  # override project if different
         "vertex_location": "us-central1",
    },

    {
        "id": "7467918709982494720",          # just the number ID
        "label": "Fine-tuned Gemini 2.5 Pro",
         "source": "google",
         "type": "finetuned",
         "modality": "text",
         "description": "Fine-tuned on our domain data",
         "vertex_project": "yorubanlp",  # override project if different
         "vertex_location": "us-central1",
    },

    # Base Gemma 3 4B — Vertex AI (no deployment needed, just use the model name)
    {
        "id": "google/gemma-3-4b-it",
        "label": "Gemma 3 4B Instruct",
        "source": "huggingface",
        "type": "base",
        "modality": "text",
        "requires_token": True,
        "description": "Google Gemma 3 4B instruction tuned",
        "endpoint_url": "https://ep54n793kczrnm3i.us-east-1.aws.endpoints.huggingface.cloud",
    },

    # Your fine-tuned Gemma 3 — HuggingFace Endpoint
    {
        "id": "Ibikemi/gemma-3-4b-african-finetuned",
        "label": "Fine-tuned Gemma 3 4b",
        "source": "huggingface",
        "type": "finetuned",
        "modality": "text",
        "requires_token": True,
        "endpoint_url": "https://your-endpoint.huggingface.cloud",  # paste your URL here
        "description": "Fine-tuned Gemma 3 on our dataset",
    },


    # ──────────────────────────────────────────────────────────────────────
    # HUGGINGFACE MODELS (run on HF Inference API — no GPU on your side)
    # Requires HF_TOKEN in your .env for private/gated models like Llama
    # ──────────────────────────────────────────────────────────────────────

    {
        "id": "NCAIR1/N-ATLaS",
        "label": "N-Atlas",
        "source": "huggingface",
        "type": "base",
        "modality": "text",
        "requires_token": True,          # needs HF_TOKEN (gated model)
        "description": "Awarri N-Atlas Text model — needs HF token approval",
    },
    {
        "id": "",
        "label": "Fine-tuned N-Atlas",
        "source": "huggingface",
        "type": "finetuned",
        "modality": "text",
        "requires_token": False,
        "description": "Finetuned N-Atlas Text model",
    },
    #{
    #    "id": "google/flan-t5-large",
    #   "label": "Flan-T5 Large",
    #    "source": "huggingface",
    #   "type": "base",
    #   "modality": "text",
    #   "requires_token": False,
    #    "description": "Google Flan-T5 Large — open access",
    #},

    # YOUR FINE-TUNED HUGGINGFACE MODEL
    # Option A: If on HF Hub (public or private repo) — uses HF Inference API
    # {
    #     "id": "your-org/your-finetuned-model",
    #     "label": "Our Fine-tuned Model v1",
    #     "source": "huggingface",
    #     "type": "finetuned",
    #     "modality": "text",
    #     "requires_token": True,         # True if private repo
    #     "description": "Fine-tuned on our dataset",
    # },

    # Option B: If deployed on HF Inference Endpoints (dedicated endpoint URL)
    # {
    #     "id": "your-org/your-finetuned-model",
    #     "label": "Our Fine-tuned Model (Endpoint)",
    #     "source": "huggingface",
    #     "type": "finetuned",
    #     "modality": "text",
    #     "requires_token": True,
    #     "endpoint_url": "https://YOUR-ENDPOINT.huggingface.cloud",   # add this
    #     "description": "Deployed on HF Inference Endpoints",
    # },

]


def get_all_models(source: str = None) -> list[dict]:
    if source:
        return [m for m in MODELS if m["source"] == source]
    return MODELS


def get_model_by_id(model_id: str) -> dict | None:
    return next((m for m in MODELS if m["id"] == model_id), None)
