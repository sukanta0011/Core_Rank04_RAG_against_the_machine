from enum import Enum
from pathlib import Path
import json
from typing import List, Dict, Any
from typing_extensions import Annotated
from pydantic import (
    TypeAdapter,
    BeforeValidator,
    AfterValidator,
    validate_call)
from langchain_text_splitters import Language
import hashlib
import os


FULL_CODE_SYNTAX = {
    'py': 'python'
}


class FileFormats(Enum):
    CODING_LANGUAGES = {item.value for item in Language}
    TXT_LANGUAGES = {'txt', 'md', 'json'}


def check_path_exists(v: str) -> str:
    if not Path(v).exists():
        raise FileNotFoundError(f"{v} does not exist")
    return v


def prepare_storage(v: str) -> str:
    path = Path(v)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        return v
    except PermissionError:
        raise PermissionError(
            f"Cannot create output file at '{path}': permission denied"
        )
    except OSError as e:
        raise OSError(f"Cannot create output path: {e}")


def prepare_storage_folder(v: str) -> str:
    path = Path(v)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return v
    except PermissionError:
        raise PermissionError(
            f"Cannot create output file at '{path}': permission denied"
        )
    except OSError as e:
        raise OSError(f"Cannot create output path: {e}")


# Define your reusable types
ExistingPath = Annotated[str, BeforeValidator(check_path_exists)]
ValidatedStoragePath = Annotated[str, AfterValidator(prepare_storage)]
PrepareStorageFolder = Annotated[str, AfterValidator(prepare_storage_folder)]


class DataManager:
    @staticmethod
    def load_data(path: ExistingPath) -> str | Dict:
        try:
            main_path = Path(path)
            if main_path.suffix == '.json':
                with open(path, 'r', errors='ignore') as fl:
                    data = json.load(fl)
                return data
            with open(path, 'r', errors='ignore') as fl:
                data = fl.read()
            return data
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Unable to read the document. Path: '{path}' does not exists."
            )
        except PermissionError:
            raise PermissionError(
                f"Unable to read the document. Path: '{path}' does not"
                " reading permission."
            )
        except Exception as e:
            raise Exception(
                f"Unknown error while trying to read '{path}'.\n"
                f"Error Message: {e}"
            )

    @staticmethod
    def generate_hash(path: ExistingPath) -> str:
        buffer_size = 65536
        hasher = hashlib.sha256()

        with open(path, "rb") as fl:
            while True:
                data = fl.read(buffer_size)
                if not data:
                    break
                hasher.update(data)
        return hasher.hexdigest()

    @staticmethod
    @validate_call
    def save_data(
        storage_path: ValidatedStoragePath,
            data: Any) -> None:
        adapter = TypeAdapter(Any)
        data_json = adapter.dump_json(
            data, indent=4).decode('utf-8')
        try:
            with open(storage_path, 'w') as fl:
                fl.write(data_json)
        except Exception as e:
            raise Exception(e)


class FilesInDir:
    @staticmethod
    def extract_all_file_paths(path: str) -> Dict[str, List[Path]]:
        parent_path = Path(path)
        if not parent_path.exists():
            raise ValueError(
                f"Directory:'{path}' does not exists"
            )

        all_paths: Dict[str, List[Path]] = {}
        only_paths: List[str] = []
        for item in parent_path.rglob("*"):
            if item.is_file():
                suffix = item.suffix[1:]
                if len(suffix) == 0:
                    continue
                if suffix in FULL_CODE_SYNTAX:
                    suffix = FULL_CODE_SYNTAX[suffix]
                try:
                    all_paths[suffix].append(item)
                except KeyError:
                    all_paths[suffix] = []
                    all_paths[suffix].append(item)

        # for syntax, paths in self._all_paths.items():
        #     print(f"{syntax}: {len(paths)}: {paths[0]}")
        only_paths = FilesInDir.get_flatten_paths(all_paths)
        FilesInDir.save_file_stats(only_paths)
        return all_paths
    
    @staticmethod
    def get_flatten_paths(paths: Dict[str, List[Path]]) -> List[Path]:
        flatten_paths = []
        for _, path in paths.items():
            flatten_paths.extend(path)
        return flatten_paths

    @staticmethod
    def get_file_stats(paths: List[Path]) -> Dict:
        stats = {}
        for path in paths:
            stats.update({
                str(path): {
                    "time": os.stat(path).st_mtime,
                    "size": os.stat(path).st_size
                }
            })
        return stats
        
    @staticmethod
    def save_file_stats(paths: List[Path]) -> None:
        stats = FilesInDir.get_file_stats(paths)
        DataManager.save_data("data/cache/all_stats.json", stats)
    
    @staticmethod
    def load_file_stats(path: str = "data/cache/all_stats.json") -> Dict:
        return DataManager.load_data(path)
    
    @staticmethod
    def has_file_stats_changed(paths: List[Path],
            saved_states_path: str = "data/cache/all_stats.json"
            ) -> bool:
        current_stats = FilesInDir.get_file_stats(paths)
        old_stats = FilesInDir.load_file_stats()
        if len(list(current_stats.keys())) != len(list(old_stats.keys())):
            return True
        for key, val in current_stats.items():
            old_val = old_stats.get(key)
            if old_val is None:
                return True
            if (val["time"] != old_val["time"]) or\
                (val["size"] != old_val["size"]):
                return True


if __name__ == "__main__":
    print(DataManager.generate_hash("src/data_retrieval/test.txt"))