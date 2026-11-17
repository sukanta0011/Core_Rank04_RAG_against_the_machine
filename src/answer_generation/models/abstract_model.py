from abc import ABC, abstractmethod
from typing import List, Dict


class Model(ABC):
    """Abstract class for answer generator"""
    @abstractmethod
    def generate_answer(self, resources: List[Dict],
                        tokens_limit: int) -> str:
        pass
