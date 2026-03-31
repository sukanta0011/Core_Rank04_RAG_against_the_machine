from abc import ABC, abstractmethod
from typing import List, Dict


class Model(ABC):
    @abstractmethod
    def generate_answer(self, resources: List[Dict],
                        tokens_limit: int) -> str:
        pass