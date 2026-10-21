from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config as C
from .common import read_open_meteo_csv
from .dq_agent import FLAG_COLUMNS, DataQualityAgent


@dataclass
class CleanResult:
    hourly: pd.DataFrame
    stations: pd.DataFrame
    decisions: pd.DataFrame
    decision_rows: pd.DataFrame
    tool_runs: pd.DataFrame
    quarantine: pd.DataFrame
    report: str


def fingerprint(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df[list(C.POLLUTANTS)], index=False).to_numpy().tobytes()
    return hashlib.sha256(h).hexdigest()[:12]


def grid_cell_id(lat: float, lon: float) -> str:
    return f"{round(float(lat), 1):.1f}N_{round(float(lon), 1):.1f}E"


def put_on_grid(df: pd.DataFrame, grid: pd.DatetimeIndex, station_id: str) -> pd.DataFrame:
    df = df.set_index("ts_utc").reindex(grid)
    df.index.name = "ts_utc"
    present = df[list(C.POLLUTANTS)].notna().any(axis=1)
    df["is_gap_row"] = (~present).to_numpy()
    for col in FLAG_COLUMNS:
        if col != "is_gap_row":
            df[col] = False
    df.insert(0, "station_id", station_id)
    return df.reset_index()


def clean_all(manifest: pd.DataFrame, run_id: str, raw_dir: Path = C.RAW) -> CleanResult:
    agent = DataQualityAgent(run_id)
    parsed: dict[str, tuple[dict, pd.DataFrame]] = {}
    for row in manifest.itertuples():
        meta, raw = read_open_meteo_csv(raw_dir / row.file)
        parsed[row.station_id] = (meta, agent.review_raw(row.station_id, meta, raw))

    start = min(d["ts_utc"].min() for _, d in parsed.values())
    end = max(d["ts_utc"].max() for _, d in parsed.values())
    grid = pd.date_range(start, end, freq="1h", tz="UTC").astype("datetime64[us, UTC]")

    frames: dict[str, pd.DataFrame] = {}
    meta_rows = []
    for row in manifest.itertuples():
        meta, df = parsed[row.station_id]
        reg = agent.review_station(row.station_id, put_on_grid(df, grid, row.station_id))
        frames[row.station_id] = reg
        meta_rows.append({
            "station_id": row.station_id, "city": row.city, "station": row.station,
            "latitude": float(meta["latitude"]), "longitude": float(meta["longitude"]),
            "elevation_m": float(meta["elevation"]), "grid_cell_id": grid_cell_id(meta["latitude"], meta["longitude"]),
            "series_group_id": fingerprint(reg), "n_rows": len(reg),
            "first_ts_utc": reg["ts_utc"].min(), "last_ts_utc": reg["ts_utc"].max(),
            "source_file": row.file, "sha256": row.sha256,
        })
    stations = agent.review_dataset(frames, pd.DataFrame(meta_rows))
    hourly = pd.concat(frames.values(), ignore_index=True).sort_values(["station_id", "ts_utc"]).reset_index(drop=True)
    return CleanResult(hourly, stations, agent.decisions_frame(), agent.decision_rows_frame(),
                       agent.tool_runs_frame(), agent.quarantine_frame(), agent.report_markdown())


def write_cleaned(result: CleanResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.hourly.to_parquet(out_dir / "aq_hourly_cleaned.parquet", index=False)
    result.stations.to_parquet(out_dir / "stations.parquet", index=False)
    result.decisions.to_csv(out_dir / "quality_log.csv", index=False)
    result.decision_rows.to_csv(out_dir / "dq_decision_rows.csv", index=False)
    result.tool_runs.to_csv(out_dir / "dq_tool_runs.csv", index=False)
    result.quarantine.to_csv(out_dir / "quarantine.csv", index=False)
    (out_dir / "dq_report.md").write_text(result.report, encoding="utf-8")
