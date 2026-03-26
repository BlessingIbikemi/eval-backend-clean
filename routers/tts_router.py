"""
tts_router.py
-------------
POST /tts/speak  — converts text to speech using YarnGPT API
                   returns audio as streamable response
"""

import os
import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/tts", tags=["Text to Speech"])

YARNGPT_URL = "https://yarngpt.ai/api/v1/tts"
YARNGPT_KEY = os.environ.get("YARNGPT_API_KEY", "")

AVAILABLE_VOICES = ["Idera", "Emma", "Zainab", "Jude", "Mary"]  # update as YarnGPT adds more


class TTSRequest(BaseModel):
    text:            str = Field(..., description="Text to convert to speech. Max 2000 characters.")
    voice:           str = Field(default="Idera", description="Voice to use")
    response_format: str = Field(default="mp3", description="Audio format: mp3, wav, opus, flac")

def truncate_to_limit(text: str, limit: int = 1800) -> str:
    """
    Truncate text to limit characters at a word boundary.
    Returns (truncated_text, was_truncated)
    """
    if len(text) <= limit:
        return text

    # Cut at limit then find last space to avoid cutting mid-word
    truncated = text[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated


@router.post("/speak")
def speak(req: TTSRequest):
    """
    Convert text to speech using YarnGPT API.
    Returns audio stream directly to the frontend.
    """
    if not YARNGPT_KEY:
        raise HTTPException(
            status_code=500,
            detail="YARNGPT_API_KEY not set. Add it to your environment variables."
        )

        # Truncate if over limit
        text = truncate_to_limit(req.text, limit=500)

    headers = {
        "Authorization": f"Bearer {YARNGPT_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "text":            req.text,
        "voice":           req.voice,
        "response_format": req.response_format,
    }

    try:
        response = requests.post(
            YARNGPT_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=120,
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="YarnGPT API timed out. Try again.")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=502, detail="Could not reach YarnGPT API.")

    if response.status_code != 200:
        try:
            error = response.json()
        except Exception:
            error = {"message": response.text}
        raise HTTPException(
            status_code=response.status_code,
            detail=f"YarnGPT API error: {error}"
        )

    # Stream audio back to frontend
    media_types = {
        "mp3":  "audio/mpeg",
        "wav":  "audio/wav",
        "opus": "audio/opus",
        "flac": "audio/flac",
    }
    media_type = media_types.get(req.response_format, "audio/mpeg")

    return StreamingResponse(
        response.iter_content(chunk_size=8192),
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename=speech.{req.response_format}"}
    )


@router.get("/voices")
def get_voices():
    """Return available voices."""
    return {"voices": AVAILABLE_VOICES}
