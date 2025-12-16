"""
File and Directory Utilities

Common file operations for experiments and data processing.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def create_output_directory(base_path: str | Path, prefix: str = "results") -> Path:
    """
    Create output directory with timestamp.

    Args:
        base_path: Base path for results
        prefix: Prefix for directory name

    Returns:
        Created directory path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_path) / f"{prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(data: Any, file_path: Path | str, indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(file_path: Path | str) -> Any:
    """
    Load data from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path) as f:
        return json.load(f)


def ensure_directory(dir_path: Path | str) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        dir_path: Directory path

    Returns:
        Directory path
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_timestamp() -> str:
    """
    Get current timestamp string.

    Returns:
        Timestamp string in YYYYMMDD_HHMMSS format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")