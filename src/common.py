from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import pandas as pd


class DataContractError(RuntimeError):
    pass


class RawIntegrityError(RuntimeError):
    pass


class ValidationError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def parse_open_meteo_text(text: str) -> tuple[dict, pd.DataFrame]:
    lines = text.lstrip("\ufeff").splitlines()
    try:
        header_idx = next(i for i, line in enumerate(lines) if line.startswith("time,"))
    except StopIteration as exc:
        raise DataContractError("no 'time,' header row found") from exc
    meta_df = pd.read_csv(io.StringIO("\n".join(lines[:header_idx])))
    if meta_df.empty:
        raise DataContractError("metadata block is empty")
    hourly = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    return meta_df.iloc[0].to_dict(), hourly


def read_open_meteo_csv(path: Path) -> tuple[dict, pd.DataFrame]:
    return parse_open_meteo_text(Path(path).read_text(encoding="utf-8"))
