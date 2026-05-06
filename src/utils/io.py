"""
BhojRAG File I/O Utilities
===========================
Helpers for reading/writing JSONL, text, TSV, and pickle files.
All functions handle encoding and path creation automatically.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file (one JSON object per line).
    
    Args:
        path: Path to the .jsonl file.
        
    Returns:
        List of parsed dictionaries.
    """
    path = Path(path)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_num}: {e}"
                ) from e
    return records


def save_jsonl(
    records: List[Dict[str, Any]],
    path: str | Path,
    append: bool = False,
) -> None:
    """
    Save records to a JSONL file.
    
    Args:
        records: List of dicts to serialize.
        path: Output file path.
        append: If True, append to existing file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_text_file(path: str | Path) -> str:
    """
    Read entire text file as a single string.
    
    Args:
        path: Path to text file.
        
    Returns:
        File contents as string.
    """
    path = Path(path)
    return path.read_text(encoding="utf-8")


def save_text_file(content: str, path: str | Path) -> None:
    """
    Write string to a text file.
    
    Args:
        content: Text content to write.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_tsv(path: str | Path) -> List[List[str]]:
    """
    Load a TSV file as list of rows (list of strings).
    
    Args:
        path: Path to TSV file.
        
    Returns:
        List of rows, each row is a list of tab-separated values.
    """
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line.split("\t"))
    return rows


def save_tsv(
    rows: List[List[str]],
    path: str | Path,
    header: Optional[List[str]] = None,
) -> None:
    """
    Save rows to a TSV file.
    
    Args:
        rows: List of rows (each row is a list of strings).
        path: Output file path.
        header: Optional header row.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")


def save_pickle(obj: Any, path: str | Path) -> None:
    """Serialize object to pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    """Deserialize object from pickle file."""
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
