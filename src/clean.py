"""Layer 2 (cleaned): typed, UTC, regularised hourly grid, every decision logged.

Decision vocabulary (from the event guide):
    FIX         certain, reversible correction (unit conversion, duplicate timestamp)
    FILL        estimate for a short interior gap, always marked `was_imputed`
    FLAG        leave as is and warn (long gaps, implausible values, identical series)
    QUARANTINE  row set aside in quarantine.csv, never silently dropped
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .common import DataContractError, read_open_meteo_csv

_COL_RE = re.compile(r"^\s*(?P<name>[A-Za-z0-9_]+)\s*\((?P<unit>[^)]*)\)\s*$")
FLAG_COLS = ["is_gap_row", "was_imputed", "qc_out_of_range", "qc_pm_inconsistent"]


@dataclass
class DecisionLog:
    """Audit trail of every fix/fill/flag/quarantine decision (-> quality_log.csv)."""
    run_id: str
    decisions: list[dict] = field(default_factory=list)
    quarantined: list[pd.DataFrame] = field(default_factory=list)

    def add(self, station_id: str, rule_id: str, action: str, variable: str, rows: int, reason: str) -> None:
        if rows <= 0:
            return
        self.decisions.append({
            "run_id": self.run_id, "decided_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "station_id": station_id, "rule_id": rule_id, "action": action, "variable": variable,
            "rows_affected": int(rows), "reason": reason,
        })

    def quarantine(self, station_id: str, rule_id: str, rows: pd.DataFrame) -> None:
        if rows.empty:
            return
        q = pd.DataFrame({"station_id": station_id, "rule_id": rule_id,
                          "raw_record": rows.astype(str).apply(lambda r: r.to_json(), axis=1)})
        self.quarantined.append(q)

    def decisions_frame(self) -> pd.DataFrame:
        cols = ["run_id", "decided_at_utc", "station_id", "rule_id", "action", "variable", "rows_affected", "reason"]
        return pd.DataFrame(self.decisions, columns=cols)

    def quarantine_frame(self) -> pd.DataFrame:
        if not self.quarantined:
            return pd.DataFrame(columns=["station_id", "rule_id", "raw_record"])
        return pd.concat(self.quarantined, ignore_index=True)


# --------------------------------------------------------------------------------------------
# Per-station steps
# --------------------------------------------------------------------------------------------
def canonicalize_columns(raw: pd.DataFrame, station_id: str, log: DecisionLog) -> pd.DataFrame:
    """Rename 'pm2_5 (μg/m³)' -> 'pm25' and convert units to μg/m³ (unknown unit = contract error)."""
    if "time" not in raw.columns:
        raise DataContractError(f"{station_id}: no 'time' column")
    out = pd.DataFrame({"time_raw": raw["time"].astype(str)})
    for col in raw.columns:
        if col == "time":
            continue
        m = _COL_RE.match(col)
        if not m or m["name"] not in C.VAR_MAP:
            raise DataContractError(f"{station_id}: unexpected column {col!r}")
        unit = m["unit"].replace("\u00b5", "\u03bc").strip()  # micro sign -> Greek mu
        if unit not in C.UNIT_TO_UG_M3:
            raise DataContractError(f"{station_id}: unknown unit {unit!r} in {col!r}")
        factor = C.UNIT_TO_UG_M3[unit]
        values = pd.to_numeric(raw[col], errors="coerce").astype("float64")
        if factor != 1.0:
            values = values * factor
            log.add(station_id, "R-UNIT", "FIX", C.VAR_MAP[m["name"]], int(values.notna().sum()),
                    f"converted {unit} to {C.CANONICAL_UNIT} (x{factor})")
        out[C.VAR_MAP[m["name"]]] = values
    for var in C.POLLUTANTS:  # a whole variable missing from the file is flagged, not invented
        if var not in out.columns:
            out[var] = np.nan
            log.add(station_id, "R-COLUMN", "FLAG", var, len(out), "variable absent from raw file; left empty")
    return out


def parse_time_and_dedupe(df: pd.DataFrame, utc_offset_s: int, station_id: str, log: DecisionLog) -> pd.DataFrame:
    """Parse timestamps to UTC; quarantine unparseable rows; keep last of duplicate timestamps."""
    naive = pd.to_datetime(df["time_raw"], format="%Y-%m-%dT%H:%M", errors="coerce")
    bad = naive.isna()
    if bad.any():
        log.quarantine(station_id, "R-TIME", df.loc[bad])
        log.add(station_id, "R-TIME", "QUARANTINE", "time", bad.sum(), "timestamp could not be parsed")
    df = df.loc[~bad].copy()
    ts = naive.loc[~bad] - pd.to_timedelta(int(utc_offset_s), unit="s")  # local -> UTC (offset is 0 here)
    df.insert(0, "ts_utc", ts.dt.tz_localize("UTC").astype("datetime64[us, UTC]"))
    df = df.drop(columns="time_raw")
    dup = df.duplicated("ts_utc", keep="last")
    if dup.any():
        log.quarantine(station_id, "R-DUP-TS", df.loc[dup])
        log.add(station_id, "R-DUP-TS", "FIX", "ts_utc", dup.sum(),
                "duplicate timestamp: kept the last occurrence, earlier copies quarantined")
    return df.loc[~dup].sort_values("ts_utc").reset_index(drop=True)


def _short_gap_mask(isna: pd.Series, max_len: int) -> pd.Series:
    """True for NaN runs no longer than max_len (whole runs only, never a partial fill)."""
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("size")
    return isna & (run_len <= max_len)


def regularize(df: pd.DataFrame, grid: pd.DatetimeIndex, station_id: str, log: DecisionLog) -> pd.DataFrame:
    """Put a station on the complete hourly grid, apply range rules, fill short gaps, add flags."""
    df = df.set_index("ts_utc").reindex(grid)
    df.index.name = "ts_utc"
    present = df[list(C.POLLUTANTS)].notna().any(axis=1)
    n_absent = int((~grid.isin(df.index[present.to_numpy()])).sum())

    is_gap_row = ~present
    log.add(station_id, "R-GAP-ROW", "FLAG", "all", int(is_gap_row.sum()),
            "hours with no data in any variable; kept as missing, not zero")

    out_of_range = pd.Series(False, index=df.index)
    for var in C.POLLUTANTS:
        neg = df[var] < 0
        if neg.any():
            df.loc[neg, var] = np.nan
            out_of_range |= neg
            log.add(station_id, "R-NEG", "FLAG", var, neg.sum(), "negative concentration is impossible; set to missing")
        high = df[var] > C.PLAUSIBLE_MAX[var]
        if high.any():
            out_of_range |= high
            log.add(station_id, "R-MAX", "FLAG", var, high.sum(),
                    f"above plausible maximum {C.PLAUSIBLE_MAX[var]}; kept, cannot be confirmed")

    pm_bad = (df["pm25"] > df["pm10"]).fillna(False)
    log.add(station_id, "R-PM-ORDER", "FLAG", "pm25", pm_bad.sum(), "PM2.5 exceeds PM10 (physically inconsistent)")

    was_imputed = pd.Series(False, index=df.index)
    for var in C.POLLUTANTS:
        isna = df[var].isna()
        if not isna.any():
            continue
        interp = df[var].interpolate(method="linear", limit_area="inside")  # interior only
        fill = _short_gap_mask(isna, C.MAX_FILL_HOURS) & interp.notna()
        if fill.any():
            df.loc[fill, var] = interp[fill]
            was_imputed |= fill
            log.add(station_id, "R-FILL", "FILL", var, fill.sum(),
                    f"interior gap <= {C.MAX_FILL_HOURS} h linearly interpolated and marked was_imputed")
        left = df[var].isna()
        log.add(station_id, "R-LONG-GAP", "FLAG", var, left.sum(), "missing after cleaning (long gap or edge); not guessed")

    df["is_gap_row"] = is_gap_row.to_numpy()
    df["was_imputed"] = was_imputed.to_numpy()
    df["qc_out_of_range"] = out_of_range.to_numpy()
    df["qc_pm_inconsistent"] = pm_bad.to_numpy()
    df.insert(0, "station_id", station_id)
    return df.reset_index()


# --------------------------------------------------------------------------------------------
# Dataset-level step
# --------------------------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """Hash of a station's pollutant series (NaN-safe, order-sensitive)."""
    h = pd.util.hash_pandas_object(df[list(C.POLLUTANTS)], index=False).to_numpy().tobytes()
    return hashlib.sha256(h).hexdigest()[:12]


def grid_cell_id(lat: float, lon: float) -> str:
    return f"{round(float(lat), 1):.1f}N_{round(float(lon), 1):.1f}E"


def flag_identical_series(stations: pd.DataFrame, log: DecisionLog) -> pd.DataFrame:
    """The supplied 'stations' are model grid cells. Several share one cell, so their series are identical.

    Keep every station in the cleaned layer (nothing deleted) but mark one primary series per group,
    so gold tables and train/test splits do not count the same numbers several times.
    """
    stations = stations.sort_values("station_id").reset_index(drop=True)
    stations["is_primary_series"] = ~stations.duplicated("series_group_id", keep="first")
    sizes = stations.groupby("series_group_id")["station_id"].transform("size")
    for _, grp in stations[sizes > 1].groupby("series_group_id"):
        dupes = grp.loc[~grp["is_primary_series"], "station_id"].tolist()
        primary = grp.loc[grp["is_primary_series"], "station_id"].iloc[0]
        log.add("ALL", "R-IDENTICAL-SERIES", "FLAG", "all", int(grp["n_rows"].iloc[0]) * len(dupes),
                f"{len(grp)} stations share one model grid cell and an identical series; primary={primary}; "
                f"excluded from gold: {', '.join(dupes)}")
    return stations


@dataclass
class CleanResult:
    hourly: pd.DataFrame
    stations: pd.DataFrame
    decisions: pd.DataFrame
    quarantine: pd.DataFrame


def clean_all(manifest: pd.DataFrame, run_id: str, raw_dir: Path = C.RAW) -> CleanResult:
    log = DecisionLog(run_id)
    parsed: dict[str, tuple[dict, pd.DataFrame]] = {}
    for row in manifest.itertuples():
        meta, raw = read_open_meteo_csv(raw_dir / row.file)
        df = canonicalize_columns(raw, row.station_id, log)
        df = parse_time_and_dedupe(df, int(meta.get("utc_offset_seconds", 0)), row.station_id, log)
        parsed[row.station_id] = (meta, df)

    # One shared window so a station that stops early shows up as a gap, not as a shorter series.
    start = min(d["ts_utc"].min() for _, d in parsed.values())
    end = max(d["ts_utc"].max() for _, d in parsed.values())
    grid = pd.date_range(start, end, freq="1h", tz="UTC").astype("datetime64[us, UTC]")

    frames, meta_rows = [], []
    for row in manifest.itertuples():
        meta, df = parsed[row.station_id]
        reg = regularize(df, grid, row.station_id, log)
        frames.append(reg)
        meta_rows.append({
            "station_id": row.station_id, "city": row.city, "station": row.station,
            "latitude": float(meta["latitude"]), "longitude": float(meta["longitude"]),
            "elevation_m": float(meta["elevation"]), "grid_cell_id": grid_cell_id(meta["latitude"], meta["longitude"]),
            "series_group_id": fingerprint(reg), "n_rows": len(reg),
            "first_ts_utc": reg["ts_utc"].min(), "last_ts_utc": reg["ts_utc"].max(),
            "source_file": row.file, "sha256": row.sha256,
        })
    stations = flag_identical_series(pd.DataFrame(meta_rows), log)
    hourly = pd.concat(frames, ignore_index=True).sort_values(["station_id", "ts_utc"]).reset_index(drop=True)
    return CleanResult(hourly, stations, log.decisions_frame(), log.quarantine_frame())


def write_cleaned(result: CleanResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.hourly.to_parquet(out_dir / "aq_hourly_cleaned.parquet", index=False)
    result.stations.to_parquet(out_dir / "stations.parquet", index=False)
    result.decisions.to_csv(out_dir / "quality_log.csv", index=False)
    result.quarantine.to_csv(out_dir / "quarantine.csv", index=False)
