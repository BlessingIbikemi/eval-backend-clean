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
        Call a fine-tuned model deployed as a Vertex AI Endpoint.
        endpoint_id: the numeric ID shown in Vertex AI → Endpoints (e.g. "1234567890123456789")
        """
        from google.cloud import aiplatform

        # Allow per-model project/location override from registry
        project  = kwargs.get("vertex_project", self.project)
        location = kwargs.get("vertex_location", self.location)

        endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
        )

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        # Vertex AI endpoint input format depends on how you deployed the model.
        # This covers the most common cases:
        instances = [{"prompt": full_prompt}]
        parameters = {
            "temperature": kwargs.get("temperature", 0.2),
            "maxOutputTokens": kwargs.get("max_tokens", 1024),
        }

        response = endpoint.predict(instances=instances, parameters=parameters)

        # Extract text from response — adjust if your endpoint returns a different structure
        prediction = response.predictions[0]
        if isinstance(prediction, dict):
            return prediction.get("content", prediction.get("output", str(prediction)))
        return str(prediction)
