"""
HuggingFace Adapter
-------------------
Handles ALL HuggingFace models without needing a local GPU:

  1. HF Serverless Inference API  — for public models and your private HF Hub models
     Free tier available. Needs HF_TOKEN for gated models (Llama etc.) and private repos.

  2. HF Dedicated Inference Endpoints — for your fine-tuned models you deployed
     as a dedicated endpoint. Provide endpoint_url in model_registry.py.

NEVER loads models locally — no GPU needed on your laptop.

Setup:
  pip install huggingface_hub
  Set HF_TOKEN in .env (get it from huggingface.co/settings/tokens)
"""

import os
from .base_adapter import BaseTextAdapter


class HuggingFaceAdapter(BaseTextAdapter):

    def __init__(self):
        self.token = os.environ.get("HF_TOKEN", "")

    def infer(self, model_id: str, prompt: str, system_prompt: str = "", **kwargs) -> str:
        # If model has a dedicated endpoint URL, use that
        endpoint_url = kwargs.get("endpoint_url")
        if endpoint_url:
            return self._call_dedicated_endpoint(prompt, system_prompt, **kwargs)

        # Otherwise use HF Serverless Inference API
        return self._call_inference_api(model_id, prompt, system_prompt, **kwargs)

    def _call_inference_api(self, model_id: str, prompt: str, system_prompt: str, **kwargs) -> str:
        """
        HuggingFace Serverless Inference API.
        Works for most public models and your private HF Hub models.
        For gated models (Llama, Mistral etc.) you need HF_TOKEN and model access approved.
        """
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

        # Try chat_completion first (works for instruct/chat models)
        # Falls back to text_generation for base models with different tokenizers
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.2),
            )
            return response.choices[0].message.content

        except Exception as chat_err:
            # Fallback: plain text generation (handles models with non-chat tokenizers)
            try:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = client.text_generation(
                    full_prompt,
                    max_new_tokens=kwargs.get("max_tokens", 512),
                    temperature=kwargs.get("temperature", 0.2),
                    return_full_text=False,
                )
                return response
            except Exception as text_err:
                raise RuntimeError(
                    f"Both chat_completion and text_generation failed for {model_id}.\n"
                    f"Chat error: {chat_err}\n"
                    f"Text error: {text_err}\n"
                    f"Check: (1) model exists on HF Hub, (2) HF_TOKEN is set if model is gated/private"
                )

    def _call_dedicated_endpoint(self, prompt: str, system_prompt: str, **kwargs) -> str:
        """
        HuggingFace Dedicated Inference Endpoints.
        Use this for your fine-tuned models deployed as paid HF endpoints.
        endpoint_url looks like: https://your-endpoint-name.us-east-1.aws.endpoints.huggingface.cloud
        """
        endpoint_url = kwargs.get("endpoint_url")
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("Run: pip install huggingface_hub")

        client = InferenceClient(
            base_url=endpoint_url,
            token=self.token if self.token else None,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.2),
            )
            return response.choices[0].message.content
        except Exception:
            # Fallback for endpoints that don't support chat format
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = client.text_generation(
                full_prompt,
                max_new_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.2),
                return_full_text=False,
            )
            return response
