from abc import ABC, abstractmethod

class BaseTextAdapter(ABC):
    @abstractmethod
    def infer(self, model_id: str, prompt: str, system_prompt: str = "", **kwargs) -> str:
        raise NotImplementedError
