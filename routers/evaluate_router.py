"""
evaluate_router.py
------------------
POST /evaluate/single  — one prompt, returns prediction + metrics
POST /evaluate/batch   — JSON list of prompts
POST /evaluate/csv     — upload a CSV file (columns: prompt, reference)
"""

import csv, io
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field

from model_registry import get_model_by_id
from adapters.adapter_router import get_adapter
from evaluators.text_evaluator import evaluate_text, aggregate

router = APIRouter(prefix="/evaluate", tags=["Evaluate"])


# ── Schemas ────────────────────────────────────────────────────────────────

class SingleRequest(BaseModel):
    model_id:      str   = Field(..., description="Model ID from the registry")
    prompt:        str   = Field(..., description="Input prompt")
    reference:     str | None     = Field(default=None, description="Ground truth / expected output (optional)")
    system_prompt: str   = Field(default="")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)

class BatchItem(BaseModel):
    prompt:    str
    reference: str

class BatchRequest(BaseModel):
    model_id:      str
    items:         list[BatchItem]
    system_prompt: str   = ""
    temperature:   float = 0.2
    max_tokens:    int   = 512


# ── Helpers ────────────────────────────────────────────────────────────────

def _resolve_model(model_id: str):
    model = get_model_by_id(model_id)
    if not model:
        raise HTTPException(404, detail=f"Model '{model_id}' not found in registry. Add it to model_registry.py.")
    return model

def _run_inference(model: dict, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
    print(f"DEBUG source: {model['source']}")
    print(f"DEBUG id: {model['id']}")
    adapter = get_adapter(model["source"])
    print(f"DEBUG adapter: {type(adapter).__name__}")
    try:
        return adapter.infer(
            model_id      = model["id"],
            prompt        = prompt,
            system_prompt = system_prompt,
            temperature   = temperature,
            max_tokens    = max_tokens,
            # Pass optional extras from registry entry to adapter
            endpoint_url    = model.get("endpoint_url"),      # HF dedicated endpoint
            vertex_project  = model.get("vertex_project"),    # Vertex override
            vertex_location = model.get("vertex_location"),
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Inference failed for {model['id']}: {str(e)}")


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/single")
def evaluate_single(req: SingleRequest):
    model      = _resolve_model(req.model_id)
    prediction = _run_inference(model, req.prompt, req.system_prompt, req.temperature, req.max_tokens)
    metrics    = evaluate_text(prediction, req.reference) if req.reference else None

    return {
        "model_id":     req.model_id,
        "model_label":  model["label"],
        "source":       model["source"],
        "type":         model["type"],
        "prompt":       req.prompt,
        "reference":    req.reference,
        "prediction":   prediction,
        "metrics":      metrics,
    }


@router.post("/batch")
def evaluate_batch(req: BatchRequest):
    model   = _resolve_model(req.model_id)
    results = []

    for item in req.items:
        try:
            prediction = _run_inference(model, item.prompt, req.system_prompt, req.temperature, req.max_tokens)
            metrics    = evaluate_text(prediction, item.reference)
            results.append({"prompt": item.prompt, "reference": item.reference,
                            "prediction": prediction, "metrics": metrics, "error": None})
        except HTTPException as e:
            results.append({"prompt": item.prompt, "reference": item.reference,
                            "prediction": None, "metrics": None, "error": e.detail})

    good = [r["metrics"] for r in results if r["metrics"]]
    return {
        "model_id": req.model_id,
        "total": len(results), "successful": len(good), "failed": len(results) - len(good),
        "aggregate_metrics": aggregate(good),
        "results": results,
    }


@router.post("/csv")
async def evaluate_csv(
    file:          UploadFile = File(...),
    model_id:      str   = Query(...),
    system_prompt: str   = Query(default=""),
    temperature:   float = Query(default=0.2),
    max_tokens:    int   = Query(default=512),
):
    """Upload a CSV with columns 'prompt' and 'reference'."""
    content = await file.read()
    try:
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    except Exception:
        raise HTTPException(400, "Could not parse CSV file.")

    if not rows:
        raise HTTPException(400, "CSV is empty.")
    if not {"prompt", "reference"}.issubset(rows[0].keys()):
        raise HTTPException(400, f"CSV must have 'prompt' and 'reference' columns. Found: {list(rows[0].keys())}")

    req = BatchRequest(
        model_id=model_id,
        items=[BatchItem(prompt=r["prompt"], reference=r["reference"]) for r in rows],
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return evaluate_batch(req)
