from pathlib import Path
from pydantic import BaseModel, field_validator
import bm25s


class Indexer(BaseModel):
    storage_path: str

    @field_validator('storage_path', mode='before')
    def validate_path(cls, path_str: str) -> str:
        path = Path(path_str)
        if path.exists():
            return path_str
        else:
            raise FileNotFoundError(f"{path} does not exists")
