"""
main.py  —  FastAPI entry point
--------------------------------
Run with:
    uvicorn main:app --reload --port 8000

API docs:
    http://localhost:8000/docs
"""
import os
import json


# ── Google credentials for cloud deployment (Render, Railway etc.) ─────────
# Locally, gcloud auth handles this automatically.
# On cloud servers, we read from environment variable instead.
creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if creds_json:
    creds_path = "/tmp/google_credentials.json"
    with open(creds_path, "w") as f:
        f.write(creds_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

from dotenv import load_dotenv
load_dotenv()  # loads .env file before anything else

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.models_router import router as models_router
from routers.evaluate_router import router as evaluate_router
from routers.tts_router import router as tts_router

app = FastAPI(
    title="Model Evaluation Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models_router)
app.include_router(evaluate_router)
app.include_router(tts_router)

@app.get("/")
def root():
    return {"status": "running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}
