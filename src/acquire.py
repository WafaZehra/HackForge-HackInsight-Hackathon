from __future__ import annotations

import argparse
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterator

import pandas as pd

from . import config as C
from .common import (RawIntegrityError, parse_open_meteo_text, sha256_bytes, sha256_file, slugify)

MANIFEST_COLUMNS = [
    "file", "city", "station", "station_id", "source", "publisher", "url_or_doi", "query", "query_note",
    "fetch_time_utc", "fetch_time_basis", "ingested_utc", "sha256", "bytes", "rows", "licence",
]


def station_identity(city_folder: str, filename: str) -> tuple[str, str, str]:
    city = C.CITY_FIXES.get(city_folder.strip(), city_folder.strip())
    stem = Path(filename).stem
    station = stem.split("_", 1)[1] if "_" in stem else stem
    station = station.strip()
    station = station[:1].upper() + station[1:]
    return city, station, f"{slugify(city)}__{slugify(station)}"


def build_query_url(lat: float, lon: float, start: str, end: str) -> str:
    vars_ = ",".join(C.API_HOURLY_VARS)
    return (f"{C.API_ENDPOINT}?latitude={lat}&longitude={lon}&hourly={vars_}"
            f"&start_date={start}&end_date={end}&timezone=GMT&format=csv")


def _iter_source(source: Path) -> Iterator[tuple[str, str, bytes, datetime]]:
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            for info in sorted(zf.infolist(), key=lambda i: i.filename):
                p = PurePosixPath(info.filename)
                if info.is_dir() or p.suffix.lower() != ".csv" or "__MACOSX" in p.parts or p.name.startswith("."):
                    continue
                ts = datetime(*info.date_time, tzinfo=timezone.utc)
                yield p.parent.name, p.name, zf.read(info), ts
    elif source.is_dir():
        for f in sorted(source.rglob("*.csv")):
            ts = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
            yield f.parent.name, f.name, f.read_bytes(), ts
    else:
        raise FileNotFoundError(f"ingest source not found: {source}")


def ingest(source: Path) -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []
    for city_folder, filename, content, file_ts in _iter_source(Path(source)):
        city, station, station_id = station_identity(city_folder, filename)
        rel = f"{city_folder}/{filename}"
        dest = C.RAW / city_folder / filename
        digest = sha256_bytes(content)
        if dest.exists() and sha256_file(dest) != digest:
            raise RawIntegrityError(f"{rel}: raw file already exists with different content; raw is immutable")
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
        meta, hourly = parse_open_meteo_text(content.decode("utf-8"))
        times = hourly["time"].astype(str)
        start, end = times.iloc[0][:10], times.iloc[-1][:10]
        rows.append({
            "file": rel, "city": city, "station": station, "station_id": station_id,
            "source": C.SOURCE_NAME, "publisher": C.PUBLISHER, "url_or_doi": C.SOURCE_URL,
            "query": build_query_url(meta["latitude"], meta["longitude"], start, end),
            "query_note": "reconstructed from file header; original request not logged",
            "fetch_time_utc": file_ts.isoformat(timespec="seconds"), "fetch_time_basis": "file timestamp",
            "ingested_utc": now, "sha256": digest, "bytes": len(content), "rows": len(hourly),
            "licence": C.LICENCE,
        })
    if not rows:
        raise FileNotFoundError(f"no CSV files found in {source}")
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values("file").reset_index(drop=True)
    manifest.to_csv(C.MANIFEST, index=False)
    return manifest


def find_incoming() -> Path | None:
    if not C.INCOMING.exists():
        return None
    zips = sorted(C.INCOMING.glob("*.zip"))
    if zips:
        return zips[0]
    return C.INCOMING if any(C.INCOMING.rglob("*.csv")) else None


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(C.MANIFEST)


def verify_raw(manifest: pd.DataFrame) -> tuple[list[str], list[str]]:
    missing, bad = [], []
    for row in manifest.itertuples():
        path = C.RAW / row.file
        if not path.exists():
            missing.append(row.file)
        elif sha256_file(path) != row.sha256:
            bad.append(row.file)
    return missing, bad


def fetch_missing(manifest: pd.DataFrame, files: list[str], *, retries: int = 3, pause_s: float = 1.0) -> None:
    import requests

    by_file = manifest.set_index("file")
    for rel in files:
        row = by_file.loc[rel]
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(row["query"], timeout=60)
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(pause_s * attempt)
        else:
            raise RawIntegrityError(f"{rel}: could not re-fetch after {retries} tries ({last_err})")
        if sha256_bytes(resp.content) != row["sha256"]:
            raise RawIntegrityError(
                f"{rel}: re-fetched file differs from manifest hash. The provider may have revised its data. "
                "Not overwriting evidence; re-ingest deliberately if you accept the new version.")
        dest = C.RAW / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        time.sleep(pause_s)


def ensure_raw() -> pd.DataFrame:
    if not C.MANIFEST.exists():
        src = find_incoming()
        if src is None:
            raise FileNotFoundError(
                "manifest.csv not found and nothing in data/incoming/. Put Data.zip in data/incoming/ "
                "(or run: python -m src.acquire ingest <path>).")
        print(f"[acquire] first run: ingesting {src}")
        ingest(src)
    manifest = load_manifest()
    missing, bad = verify_raw(manifest)
    if bad:
        raise RawIntegrityError(f"{len(bad)} raw file(s) changed since manifest was written: {bad[:3]}")
    if missing:
        print(f"[acquire] {len(missing)} raw file(s) missing; re-fetching from manifest queries")
        fetch_missing(manifest, missing)
        missing, bad = verify_raw(manifest)
        if missing or bad:
            raise RawIntegrityError(f"raw verification failed after fetch: missing={missing[:3]} bad={bad[:3]}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.acquire")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="copy supplied CSVs into data/raw and write manifest.csv")
    p_ing.add_argument("source", type=Path)
    sub.add_parser("verify", help="re-hash raw files against manifest.csv")
    args = ap.parse_args(argv)
    if args.cmd == "ingest":
        m = ingest(args.source)
        print(f"ingested {len(m)} files into {C.RAW} ; manifest -> {C.MANIFEST}")
        return 0
    missing, bad = verify_raw(load_manifest())
    print(f"missing={len(missing)} hash_mismatch={len(bad)}")
    return 0 if not (missing or bad) else 1


if __name__ == "__main__":
    sys.exit(main())
