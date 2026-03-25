from enum import Enum
from pathlib import Path
from typing import List, Dict
from pydantic import TypeAdapter
from langchain_text_splitters import Language


FULL_CODE_SYNTAX = {
    'py': 'python'
}


class FileFormats(Enum):
    CODING_LANGUAGES = {item.value for item in Language}
    TXT_LANGUAGES = {'txt', 'md', 'json'}


class DataManager:
    @staticmethod
    def load_data(path: str) -> str:
        try:
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
    def save_data(storage_path: str, data: List[str]) -> None:
        adapter = TypeAdapter(List[str])
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
        return all_paths
