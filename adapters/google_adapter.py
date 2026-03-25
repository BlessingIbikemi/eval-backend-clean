"""
Google Vertex AI Adapter
------------------------
Handles:
  - Gemini 2.5 Pro / Flash (base models via model name)
  - Your fine-tuned models deployed as Vertex AI Endpoints (via endpoint ID)

No GPU needed — all inference runs on Google's cloud.

Setup:
  1. pip install google-cloud-ai platform
  2. Run: gcloud auth application-default login
     OR set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json in .env
  3. Set GOOGLE_CLOUD_PROJECT in .env
"""

import os
from .base_adapter import BaseTextAdapter


class GoogleAdapter(BaseTextAdapter):

    def __init__(self):
        self.project  = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        self._init_done = False

    def _init(self):
        if self._init_done:
            return
        try:
            import vertexai
            vertexai.init(project=self.project, location=self.location)
            self._init_done = True
        except ImportError:
            raise ImportError("Run: pip install google-cloud-aiplatform")

    def infer(self, model_id: str, prompt: str, system_prompt: str = "", **kwargs) -> str:
        self._init()

        # Check if this is a fine-tuned endpoint (numeric ID) or a base Gemini model name
        is_endpoint = model_id.strip().isdigit() or model_id.startswith("projects/")

        if is_endpoint:
            return self._call_finetuned_endpoint(model_id, prompt, system_prompt, **kwargs)
        else:
            return self._call_gemini(model_id, prompt, system_prompt, **kwargs)

    def _call_gemini(self, model_id: str, prompt: str, system_prompt: str, **kwargs) -> str:
        """Call Gemini 2.5 Pro / Flash base models."""
        from vertexai.generative_models import GenerativeModel, GenerationConfig
        print("I want to load the model")

        model = GenerativeModel(
            model_id,
            system_instruction=system_prompt if system_prompt else None,
        )
        print("I have loaded the model")
        config = GenerationConfig(
            temperature=kwargs.get("temperature", 0.2),
            max_output_tokens=kwargs.get("max_tokens", 1024),
        )
        response = model.generate_content(prompt)

        print("I have gotten my response")
        return response.text

    def _call_finetuned_endpoint(self, endpoint_id: str, prompt: str, system_prompt: str, **kwargs) -> str:
        """
        Call a fine-tuned Gemini model deployed on Vertex AI.
        Uses the Gemini GenerativeModel API not the standard predict endpoint.
        """
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        project = kwargs.get("vertex_project", self.project)
        location = kwargs.get("vertex_location", self.location)

        # Re-initialise vertexai with the correct project/location
        # in case it differs from the default
        import vertexai
        vertexai.init(project=project, location=location)

        # Fine-tuned Gemini models are accessed via their full resource name
        full_model_name = (
            f"projects/{project}/locations/{location}"
            f"/endpoints/{endpoint_id}"
        )

        model = GenerativeModel(
            full_model_name,
            system_instruction=system_prompt if system_prompt else None,
        )
        print("I have loaded my model")

        config = GenerationConfig(
            temperature=kwargs.get("temperature", 0.2),
            max_output_tokens=kwargs.get("max_tokens", 1024),
        )

        response = model.generate_content(prompt)
        print("I have gotten my response")
        return response.text
