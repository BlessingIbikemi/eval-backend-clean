"""
HuggingFace Adapter
-------------------
Handles ALL HuggingFace models without needing a local GPU:

  1. HF Serverless Inference API  — for public models and your private HF Hub models
  2. HF Dedicated Inference Endpoints — for your fine-tuned/deployed models

Setup:
  pip install huggingface_hub
  Set HF_TOKEN in .env
"""

import os
from .base_adapter import BaseTextAdapter


class HuggingFaceAdapter(BaseTextAdapter):

    def __init__(self):
        self.token = os.environ.get("HF_TOKEN", "")

    def infer(self, model_id: str, prompt: str, system_prompt: str = "", **kwargs) -> str:
        endpoint_url = kwargs.get("endpoint_url")
        if endpoint_url:
            return self._call_dedicated_endpoint(prompt, system_prompt, **kwargs)
        return self._call_inference_api(model_id, prompt, system_prompt, **kwargs)

    def _call_inference_api(self, model_id: str, prompt: str, system_prompt: str, **kwargs) -> str:
        """HuggingFace Serverless Inference API."""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("Run: pip install huggingface_hub")

        client = InferenceClient(
            model=model_id,
            token=self.token if self.token else None,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 2048),
                temperature=kwargs.get("temperature", 0.2),
            )
            return response.choices[0].message.content

        except Exception as chat_err:
            try:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = client.text_generation(
                    full_prompt,
                    max_new_tokens=kwargs.get("max_tokens", 2048),
                    temperature=kwargs.get("temperature", 0.2),
                    return_full_text=False,
                )
                return response
            except Exception as text_err:
                raise RuntimeError(
                    f"Both chat_completion and text_generation failed for {model_id}.\n"
                    f"Chat error: {chat_err}\nText error: {text_err}"
                )

    def _call_dedicated_endpoint(self, prompt: str, system_prompt: str, **kwargs) -> str:
        """
        HuggingFace Dedicated Inference Endpoints.

        Tries strategies in order based on what the endpoint supports.
        You can skip the guessing entirely by setting 'task' in model_registry.py:
          task: "chat"        → uses /v1/chat/completions  (Gemma, Llama, Mistral TGI)
          task: "generation"  → uses {"inputs": "..."}     (custom text generation handler)
          task: "qa"          → uses {"inputs": {"question": ..., "context": ...}}
        If task is not set, all strategies are tried automatically.
        """
        import requests

        endpoint_url = kwargs.get("endpoint_url").rstrip("/")
        full_prompt  = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        max_tokens   = kwargs.get("max_tokens", 1000000)
        temperature  = kwargs.get("temperature", 0.2)
        task         = kwargs.get("task", None)  # optional hint from model_registry

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        errors = {}

        # ── Strategy 1: OpenAI-compatible /v1/chat/completions (TGI) ──────────
        # For: Gemma, Llama, Mistral, Falcon etc. on standard TGI dedicated endpoints
        if task in (None, "chat"):
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model":       "tgi",
                    "messages":    messages,
                    "max_tokens":  max_tokens,
                    "temperature": temperature,
                }
                resp = requests.post(
                    f"{endpoint_url}/v1/chat/completions",
                    headers=headers, json=payload, timeout=120,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                errors["chat(/v1/chat/completions)"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                errors["chat(/v1/chat/completions)"] = str(e)

        # ── Strategy 2: {"inputs": "..."} — custom text generation handler ────
        # For: fine-tuned models with a simple handler.py
        if task in (None, "generation"):
            try:
                payload = {"inputs": full_prompt}
                resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 200:
                    return self._extract_text(resp.json(), full_prompt)
                errors["generation({inputs: str})"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                errors["generation({inputs: str})"] = str(e)

        # ── Strategy 3: QA pipeline ───────────────────────────────────────────
        # For: endpoints deployed as question-answering pipelines
        # Treats system_prompt as context, prompt as the question
        # If no system_prompt, uses prompt as both question and context
        if task in (None, "qa"):
            try:
                question = prompt
                context  = system_prompt if system_prompt else prompt
                payload  = {"inputs": {"question": question, "context": context}}
                resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 200:
                    result = resp.json()
                    # QA pipeline returns {"answer": "...", "score": ..., "start": ..., "end": ...}
                    if isinstance(result, dict) and "answer" in result:
                        return result["answer"]
                    return self._extract_text(result, full_prompt)
                errors["qa({question, context})"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                errors["qa({question, context})"] = str(e)

        # ── All failed ────────────────────────────────────────────────────────
        error_detail = "\n".join(f"  [{k}] {v}" for k, v in errors.items())
        raise RuntimeError(
            f"All endpoint strategies failed for {endpoint_url}.\n"
            f"Errors:\n{error_detail}\n\n"
            f"TIP: Add 'task' to this model in model_registry.py to skip guessing:\n"
            f"  task='chat'        for TGI endpoints (Gemma, Llama etc.)\n"
            f"  task='generation'  for custom handler.py text generation\n"
            f"  task='qa'          for question-answering pipeline endpoints"
        )

    def _extract_text(self, result, full_prompt: str) -> str:
        """Parse common handler response shapes into a plain string."""
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict) and "generated_text" in first:
                text = first["generated_text"]
                if text.startswith(full_prompt):
                    text = text[len(full_prompt):].strip()
                return text
            if isinstance(first, str):
                return first

        if isinstance(result, dict):
            for key in ("generated_text", "text", "response", "output", "answer"):
                if key in result:
                    return result[key]

        return str(result)
