"""File-system helpers: path validation, file loading, hashing, saving."""
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


FULL_CODE_SYNTAX = {
    'py': 'python'
}


class FileFormats(Enum):
    """File extensions the indexer knows how to chunk.

    Attributes:
        CODING_LANGUAGES: Language names supported by the code splitter.
        TXT_LANGUAGES: Extensions handled by the plain-text splitter.
    """

    CODING_LANGUAGES = {item.value for item in Language}
    TXT_LANGUAGES = {'txt', 'md', 'json'}


def check_path_exists(v: str) -> str:
    """Validate that a path exists.

    Args:
        v: Path to check.

    Returns:
        The same path, unchanged.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not Path(v).exists():
        raise FileNotFoundError(f"{v} does not exist")
    return v


def prepare_storage(v: str) -> str:
    """Make sure a file path can be written to.

    Creates the parent folders, and creates an empty file if it does
    not exist yet.

    Args:
        v: Path of the file to prepare.

    Returns:
        The same path, unchanged.

    Raises:
        IsADirectoryError: If the path is an existing directory.
        PermissionError: If the folder or file cannot be created.
        OSError: For any other file-system failure.
    """
    path = Path(v)
    try:
        if path.is_dir():
            raise IsADirectoryError(
                f"'{path}' is a directory, a file path is required")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        return v
    except IsADirectoryError:
        raise
    except PermissionError as e:
        raise PermissionError(
            f"Cannot create output file at '{path}': permission denied"
        ) from e
    except OSError as e:
        raise OSError(f"Cannot create output path '{path}': {e}") from e


def prepare_storage_folder(v: str) -> str:
    """Make sure a folder exists, creating it if needed.

    Args:
        v: Path of the folder to prepare.

    Returns:
        The same path, unchanged.

    Raises:
        PermissionError: If the folder cannot be created.
        OSError: For any other failure, e.g. the path is a file.
    """
    path = Path(v)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return v
    except PermissionError as e:
        raise PermissionError(
            f"Cannot create output folder at '{path}': permission denied"
        ) from e
    except OSError as e:
        raise OSError(f"Cannot create output folder '{path}': {e}") from e


# Reusable pydantic types
ExistingPath = Annotated[str, BeforeValidator(check_path_exists)]
ValidatedStoragePath = Annotated[str, AfterValidator(prepare_storage)]
PrepareStorageFolder = Annotated[str, AfterValidator(prepare_storage_folder)]


class DataManager:
    """Static helpers to read, hash and save files."""

    @staticmethod
    def load_data(path: ExistingPath) -> str | Dict:
        """Read a file as UTF-8 without newline translation.

        `\\r\\n` is kept as it is in the file, so character offsets match
        the real file.

        Args:
            path: File to read.

        Returns:
            For `.json` files, the parsed JSON (a dict or list, not the
            file text). For every other file, the raw text. Use
            `load_text` when you need the text of a `.json` file.

        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read.
            ValueError: If a `.json` file is not valid JSON, or the file
                is not valid UTF-8.
            Exception: For any other failure.
        """
        try:
            main_path = Path(path)
            if main_path.suffix == '.json':
                with open(path, 'r', encoding="utf-8", newline="") as fl:
                    dict_data: Dict[Any, Any] = json.load(fl)
                return dict_data
            with open(path, 'r', encoding="utf-8", newline="") as fl:
                txt_data: str = fl.read()
            return txt_data
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Unable to read the document. Path: '{path}' "
                "does not exist."
            ) from e
        except IsADirectoryError as e:
            raise IsADirectoryError(
                f"Unable to read '{path}': it is a directory."
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Unable to read the document. Path: '{path}': "
                "no read permission."
            ) from e
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(
                f"'{path}' is not valid JSON or not valid UTF-8: {e}"
            ) from e
        except Exception as e:
            raise Exception(
                f"Unknown error while trying to read '{path}'.\n"
                f"Error Message: {e}"
            ) from e

    @staticmethod
    def load_text(path: ExistingPath) -> str:
        """Read any file as raw UTF-8 text, never parsing JSON.

        Unlike `load_data`, a `.json` file comes back as its real
        text, so character offsets match the file on disk.

        Args:
            path: File to read.

        Returns:
            The file content, with `\\r\\n` kept as it is.

        Raises:
            ValueError: If the file is not valid UTF-8.
            OSError: If the file cannot be read.
        """
        try:
            with open(path, 'r', encoding="utf-8", newline="") as fl:
                return fl.read()
        except UnicodeDecodeError as e:
            raise ValueError(
                f"'{path}' is not valid UTF-8: {e}") from e

    @staticmethod
    def generate_hash(path: ExistingPath) -> str:
        """Compute the SHA-256 of a file's bytes.

        The file is read in binary mode, in blocks, so no decoding can
        fail and large files do not fill the memory.

        Args:
            path: File to hash.

        Returns:
            The hex digest.
        """
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
    def generate_hash_from_text(
            text: str) -> str:
        """Compute the SHA-256 of a string (encoded as UTF-8).

        Args:
            text: Text to hash.

        Returns:
            The hex digest.
        """
        hasher = hashlib.sha256()
        hasher.update(text.encode('utf-8'))
        return hasher.hexdigest()

    @staticmethod
    @validate_call
    def save_data(
        storage_path: ValidatedStoragePath,
            data: Any, method: str = "w") -> None:
        """Serialize `data` as indented JSON and write it to a file.

        The parent folders (and an empty file) are created first by the
        `ValidatedStoragePath` validator. Pydantic models, dicts, lists
        and plain values are all accepted.

        Args:
            storage_path: Destination file.
            data: Object to serialize.
            method: File mode, "w" (overwrite) by default. Appending
                ("a") would produce several JSON documents in one file.

        Raises:
            Exception: If serialization or writing fails. The original
                error is chained as the cause.
        """
        adapter = TypeAdapter(Any)
        data_json = adapter.dump_json(
            data, indent=4).decode('utf-8')
        try:
            with open(
                storage_path, method,
                    encoding="utf-8", newline="") as fl:
                fl.write(data_json)
        except Exception as e:
            raise Exception(
                f"Unable to write '{storage_path}': {e}") from e


class FilesInDir:
    """Helpers to list the files of a folder."""

    @staticmethod
    def extract_all_file_paths(path: str) -> Dict[str, List[Path]]:
        """Group every file under `path` by extension.

        Files without an extension are skipped. The `py` extension is
        stored under `python`, the name the code splitter expects. Paths
        are sorted, so the result does not depend on the order in which
        the operating system lists files.

        Args:
            path: Folder to scan (recursively).

        Returns:
            A dict mapping an extension (or language name) to the list
            of files that have it.

        Raises:
            ValueError: If the path does not exist or is not a folder.
        """
        parent_path = Path(path)
        if not parent_path.exists():
            raise ValueError(
                f"Directory:'{path}' does not exists"
            )
        if not parent_path.is_dir():
            raise ValueError(f"'{path}' is not a directory")

        all_paths: Dict[str, List[Path]] = {}
        for item in sorted(parent_path.rglob("*")):
            if item.is_file():
                suffix = item.suffix[1:]
                if len(suffix) == 0:
                    continue
                if suffix in FULL_CODE_SYNTAX:
                    suffix = FULL_CODE_SYNTAX[suffix]
                all_paths.setdefault(suffix, []).append(item)
        return all_paths


if __name__ == "__main__":
    print(DataManager.generate_hash("src/data_retrieval/test.txt"))
