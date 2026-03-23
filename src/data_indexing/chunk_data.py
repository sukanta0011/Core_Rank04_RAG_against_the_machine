from pydantic import BaseModel, Field
from typing import List


class MinimalSource(BaseModel):
    """The MinimalSource model represents a minimal source of information"""
    file_path: str
    first_character_index: int
    last_character_index: int


class SplitDataByChunks(BaseModel):
    def __init__(self) -> None:
        pass
