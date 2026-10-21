from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C
from .common import DataContractError

FIX, FILL, FLAG, QUARANTINE = "FIX", "FILL", "FLAG", "QUARANTINE"
FLAG_COLUMNS = ["is_gap_row", "was_imputed", "qc_out_of_range", "qc_pm_inconsistent", "qc_spike"]
COLUMN_PATTERN = re.compile(r"^\s*(?P<name>[A-Za-z0-9_]+)\s*\((?P<unit>[^)]*)\)\s*$")
NO_ROWS = np.array([], dtype=int)

POLICY = {
    "R-UNIT": (FIX, "known unit with an exact conversion: certain and reversible"),
    "R-UTC-OFFSET": (FIX, "known offset to UTC: certain and reversible"),
    "R-DUP-TS": (FIX, "later record supersedes the earlier one; earlier copy is quarantined, not lost"),
    "R-TIME-PARSE": (QUARANTINE, "row cannot be placed on the time axis, so it is set aside rather than guessed"),
    "R-VAR-MISSING": (FLAG, "a whole variable is absent; left empty instead of invented"),
    "R-NEG": (FLAG, "impossible value that cannot be corrected without guessing; set to missing and marked"),
    "R-MAX": (FLAG, "implausible value that cannot be confirmed; kept and marked"),
    "R-PM-ORDER": (FLAG, "physically inconsistent pair; cannot tell which value is wrong"),
    "R-GAP-FILL": (FILL, "short interior gap in a smooth series: a reasonable estimate, kept visible via was_imputed"),
    "R-GAP-EDGE": (FLAG, "gap touches the start or end of the series; nothing to interpolate between"),
    "R-GAP-LONG": (FLAG, "gap longer than the fill limit; guessing would hide real uncertainty"),
    "R-GAP-JUMP": (FLAG, "short gap but the series changes too fast across it to estimate safely"),
    "R-SPIKE": (FLAG, "extreme but possibly real episode; never removed automatically"),
    "R-IDENTICAL-SERIES": (FLAG, "duplicate evidence of one model grid cell; all rows kept, one series marked primary"),
    "R-STALE": (FLAG, "source stopped updating earlier than its peers; users must see that it is old"),
    "R-DRIFT": (FLAG, "distribution changed between the first and second half of the record; model may not transfer"),
}


@dataclass
class Finding:
    tool: str
    rule_id: str
    scope: str
    station_id: str
    variable: str
    positions: np.ndarray
    evidence: dict = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(len(self.positions))


def parse_column(col: str) -> tuple[str, str] | None:
    m = COLUMN_PATTERN.match(col)
    if not m or m["name"] not in C.VAR_MAP:
        return None
    return C.VAR_MAP[m["name"]], m["unit"].replace("\u00b5", "\u03bc").strip()


def make_row_ids(station_id: str, ts: pd.Series) -> list[str]:
    return (station_id + ":" + ts.dt.strftime("%Y%m%d%H")).tolist()


def tool_columns(station_id: str, columns: list[str]) -> list[Finding]:
    findings, seen = [], set()
    for col in columns:
        if col == "time":
            continue
        parsed = parse_column(col)
        if parsed is None:
            findings.append(Finding("tool_columns", "R-COLUMN-UNEXPECTED", "file", station_id, col, NO_ROWS, {"column": col}))
            continue
        var, unit = parsed
        seen.add(var)
        if unit not in C.UNIT_TO_UG_M3:
            findings.append(Finding("tool_columns", "R-UNIT-UNKNOWN", "file", station_id, var, NO_ROWS, {"column": col, "unit": unit}))
        elif C.UNIT_TO_UG_M3[unit] != 1.0:
            findings.append(Finding("tool_columns", "R-UNIT", "file", station_id, var, NO_ROWS,
                                    {"column": col, "unit": unit, "factor": C.UNIT_TO_UG_M3[unit]}))
    for var in C.POLLUTANTS:
        if var not in seen:
            findings.append(Finding("tool_columns", "R-VAR-MISSING", "file", station_id, var, NO_ROWS, {}))
    return findings


def tool_timestamps(station_id: str, time_raw: pd.Series, naive: pd.Series, offset_s: int) -> list[Finding]:
    findings = []
    bad = naive.isna().to_numpy()
    if bad.any():
        findings.append(Finding("tool_timestamps", "R-TIME-PARSE", "file", station_id, "time", np.flatnonzero(bad),
                                {"examples": time_raw[bad].head(3).tolist()}))
    if offset_s != 0:
        findings.append(Finding("tool_timestamps", "R-UTC-OFFSET", "file", station_id, "time", np.flatnonzero(~bad),
                                {"offset_s": int(offset_s)}))
    ts = naive - pd.to_timedelta(int(offset_s), unit="s")
    dup = (ts.duplicated(keep="last") & ts.notna()).to_numpy()
    if dup.any():
        findings.append(Finding("tool_timestamps", "R-DUP-TS", "file", station_id, "time", np.flatnonzero(dup),
                                {"examples": ts[dup].head(3).dt.strftime("%Y-%m-%dT%H:%M").tolist()}))
    return findings


def tool_range(station_id: str, df: pd.DataFrame) -> list[Finding]:
    findings = []
    for var in C.POLLUTANTS:
        values = df[var].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            neg = np.flatnonzero(values < 0)
            high = np.flatnonzero(values > C.PLAUSIBLE_MAX[var])
        if len(neg):
            findings.append(Finding("tool_range", "R-NEG", "station", station_id, var, neg, {"min": float(values[neg].min())}))
        if len(high):
            findings.append(Finding("tool_range", "R-MAX", "station", station_id, var, high,
                                    {"max": float(values[high].max()), "limit": C.PLAUSIBLE_MAX[var]}))
    return findings


def tool_pm_order(station_id: str, df: pd.DataFrame) -> list[Finding]:
    with np.errstate(invalid="ignore"):
        bad = np.flatnonzero((df["pm25"].to_numpy(dtype=float) > df["pm10"].to_numpy(dtype=float)))
    if not len(bad):
        return []
    return [Finding("tool_pm_order", "R-PM-ORDER", "station", station_id, "pm25", bad, {})]


def nan_runs(values: np.ndarray) -> pd.DataFrame:
    isn = np.isnan(values)
    n = len(values)
    if not isn.any():
        return pd.DataFrame(columns=["start", "end", "length", "edge", "jump", "allowed"])
    edges = np.diff(np.concatenate(([0], isn.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    length = ends - starts
    edge = (starts == 0) | (ends == n)
    before = np.where(starts > 0, values[np.maximum(starts - 1, 0)], np.nan)
    after = np.where(ends < n, values[np.minimum(ends, n - 1)], np.nan)
    steps = np.abs(np.diff(values))
    steps = steps[~np.isnan(steps)]
    typical = float(np.quantile(steps, C.SMOOTH_QUANTILE)) if len(steps) else 0.0
    return pd.DataFrame({"start": starts, "end": ends, "length": length, "edge": edge,
                         "jump": np.abs(after - before), "allowed": typical * (length + 1)})


def tool_gaps(station_id: str, df: pd.DataFrame) -> list[Finding]:
    findings = []
    for var in C.POLLUTANTS:
        runs = nan_runs(df[var].to_numpy(dtype=float))
        if runs.empty:
            continue
        positions = np.concatenate([np.arange(s, e) for s, e in zip(runs["start"], runs["end"])])
        findings.append(Finding("tool_gaps", "R-GAP", "station", station_id, var, positions, {"runs": runs}))
    return findings


def tool_spikes(station_id: str, df: pd.DataFrame) -> list[Finding]:
    findings = []
    for var in C.POLLUTANTS:
        s = df[var].astype(float)
        median = s.rolling(C.SPIKE_WINDOW, center=True, min_periods=C.SPIKE_WINDOW // 2 + 1).median()
        resid = (s - median).to_numpy()
        if np.isnan(resid).all():
            continue
        sigma = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        if sigma <= 0:
            continue
        with np.errstate(invalid="ignore"):
            hits = np.flatnonzero(resid > C.SPIKE_K * sigma)
        if len(hits):
            findings.append(Finding("tool_spikes", "R-SPIKE", "station", station_id, var, hits,
                                    {"max": float(s.to_numpy()[hits].max()), "threshold": float(C.SPIKE_K * sigma),
                                     "sigma": float(sigma)}))
    return findings


def tool_identical_series(station_id: str, stations: pd.DataFrame) -> list[Finding]:
    findings = []
    sizes = stations.groupby("series_group_id")["station_id"].transform("size")
    for gid, grp in stations[sizes > 1].groupby("series_group_id"):
        members = sorted(grp["station_id"])
        findings.append(Finding("tool_identical_series", "R-IDENTICAL-SERIES", "dataset", "ALL", "all", NO_ROWS,
                                {"group": gid, "members": members, "n_rows": int(grp["n_rows"].iloc[0]),
                                 "cell": grp["grid_cell_id"].iloc[0]}))
    return findings


def tool_stale_sources(station_id: str, frames: dict[str, pd.DataFrame]) -> list[Finding]:
    last = {sid: df.loc[df["pm25"].notna(), "ts_utc"].max() for sid, df in frames.items()}
    valid = [v for v in last.values() if pd.notna(v)]
    if not valid:
        return []
    newest = max(valid)
    findings = []
    for sid, ts in last.items():
        lag = None if pd.isna(ts) else (newest - ts).total_seconds() / 3600
        if lag is None or lag > C.STALE_GRACE_HOURS:
            df = frames[sid]
            tail = np.flatnonzero((df["ts_utc"] > ts).to_numpy()) if pd.notna(ts) else np.arange(len(df))
            findings.append(Finding("tool_stale_sources", "R-STALE", "dataset", sid, "pm25", tail,
                                    {"last_valid": None if pd.isna(ts) else ts.isoformat(), "newest": newest.isoformat(),
                                     "lag_hours": lag}))
    return findings


def psi(a: np.ndarray, b: np.ndarray) -> float | None:
    edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def tool_distribution_shift(station_id: str, frames: dict[str, pd.DataFrame]) -> list[Finding]:
    findings = []
    for sid, df in frames.items():
        half = len(df) // 2
        for var in C.POLLUTANTS:
            a = df[var].iloc[:half].dropna().to_numpy(dtype=float)
            b = df[var].iloc[half:].dropna().to_numpy(dtype=float)
            if len(a) < C.PSI_MIN_OBS or len(b) < C.PSI_MIN_OBS:
                continue
            value = psi(a, b)
            if value is not None and value > C.PSI_THRESHOLD:
                findings.append(Finding("tool_distribution_shift", "R-DRIFT", "dataset", sid, var, NO_ROWS,
                                        {"psi": value, "mean_first": float(a.mean()), "mean_second": float(b.mean())}))
    return findings


class DataQualityAgent:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.decisions: list[dict] = []
        self.decision_rows: list[dict] = []
        self.tool_runs: list[dict] = []
        self.quarantined: list[pd.DataFrame] = []

    def _run(self, tool, scope: str, station_id: str, *args) -> list[Finding]:
        findings = tool(station_id, *args)
        self.tool_runs.append({"tool": tool.__name__, "scope": scope, "station_id": station_id,
                               "findings": len(findings), "rows_flagged": int(sum(f.n_rows for f in findings))})
        return findings

    def _log(self, f: Finding, rule_id: str, explanation: str, rows: int, row_ids: list[str], evidence: dict | None = None) -> None:
        action, policy = POLICY[rule_id]
        decision_id = f"{self.run_id}-D{len(self.decisions) + 1:05d}"
        self.decisions.append({
            "decision_id": decision_id, "run_id": self.run_id,
            "decided_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scope": f.scope, "station_id": f.station_id, "tool": f.tool, "rule_id": rule_id, "action": action,
            "variable": f.variable, "rows_affected": int(rows), "policy": policy, "explanation": explanation,
            "evidence": json.dumps(evidence if evidence is not None else
                                   {k: v for k, v in f.evidence.items() if k != "runs"}, default=str),
            "example_row_ids": ";".join(row_ids[:5]),
        })
        self.decision_rows.extend({"decision_id": decision_id, "row_id": r} for r in row_ids)

    def _quarantine(self, station_id: str, rule_id: str, rows: pd.DataFrame) -> None:
        if rows.empty:
            return
        self.quarantined.append(pd.DataFrame({
            "station_id": station_id, "rule_id": rule_id,
            "raw_record": rows.astype(str).apply(lambda r: r.to_json(), axis=1)}))

    def review_raw(self, station_id: str, meta: dict, raw: pd.DataFrame) -> pd.DataFrame:
        if "time" not in raw.columns:
            raise DataContractError(f"{station_id}: no 'time' column")
        factors: dict[str, float] = {}
        for f in self._run(tool_columns, "file", station_id, list(raw.columns)):
            if f.rule_id in ("R-COLUMN-UNEXPECTED", "R-UNIT-UNKNOWN"):
                raise DataContractError(
                    f"{station_id}: {f.rule_id} {f.evidence}. No safe fix exists, so the run is stopped instead of guessing.")
            if f.rule_id == "R-UNIT":
                factors[f.variable] = f.evidence["factor"]
                self._log(f, "R-UNIT",
                          f"Column {f.evidence['column']} is reported in {f.evidence['unit']}, a known unit. Values were "
                          f"multiplied by {f.evidence['factor']} to express them in {C.CANONICAL_UNIT}.",
                          int(raw[f.evidence["column"]].notna().sum()), [])
            elif f.rule_id == "R-VAR-MISSING":
                self._log(f, "R-VAR-MISSING",
                          f"Variable {f.variable} does not exist in this file. The column was left empty; no values were invented.",
                          len(raw), [])
        frame = pd.DataFrame({"time_raw": raw["time"].astype(str)})
        for col in raw.columns:
            parsed = parse_column(col) if col != "time" else None
            if parsed:
                frame[parsed[0]] = pd.to_numeric(raw[col], errors="coerce").astype("float64") * factors.get(parsed[0], 1.0)
        for var in C.POLLUTANTS:
            if var not in frame.columns:
                frame[var] = np.nan
        naive = pd.to_datetime(frame["time_raw"], format="%Y-%m-%dT%H:%M", errors="coerce")
        offset = int(meta.get("utc_offset_seconds", 0))
        drop = np.zeros(len(frame), dtype=bool)
        for f in self._run(tool_timestamps, "file", station_id, frame["time_raw"], naive, offset):
            if f.rule_id == "R-TIME-PARSE":
                self._quarantine(station_id, f.rule_id, frame.iloc[f.positions])
                drop[f.positions] = True
                self._log(f, f.rule_id, f"{f.n_rows} timestamp(s) could not be parsed (for example {f.evidence['examples']}). "
                          "The rows cannot be placed in time, so they were moved to quarantine.csv instead of being dropped silently.",
                          f.n_rows, [])
            elif f.rule_id == "R-UTC-OFFSET":
                self._log(f, f.rule_id, f"The file declares a UTC offset of {f.evidence['offset_s']} s. Timestamps were shifted to UTC.",
                          f.n_rows, [])
            elif f.rule_id == "R-DUP-TS":
                self._quarantine(station_id, f.rule_id, frame.iloc[f.positions])
                drop[f.positions] = True
                ts = (naive - pd.to_timedelta(offset, unit="s")).iloc[f.positions]
                self._log(f, f.rule_id, f"{f.n_rows} timestamp(s) appear more than once (for example {f.evidence['examples']}). "
                          "The last record was kept as the correction and the earlier copies were moved to quarantine.csv.",
                          f.n_rows, make_row_ids(station_id, ts))
        ts_utc = (naive - pd.to_timedelta(offset, unit="s")).dt.tz_localize("UTC").astype("datetime64[us, UTC]")
        out = frame.drop(columns="time_raw")
        out.insert(0, "ts_utc", ts_utc)
        return out.loc[~drop].sort_values("ts_utc").reset_index(drop=True)

    def review_station(self, station_id: str, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ids = lambda pos: make_row_ids(station_id, df["ts_utc"].iloc[pos])

        for f in self._run(tool_range, "station", station_id, df):
            arr = df[f.variable].to_numpy(dtype=float, copy=True)
            flags = df["qc_out_of_range"].to_numpy(copy=True)
            flags[f.positions] = True
            df["qc_out_of_range"] = flags
            if f.rule_id == "R-NEG":
                arr[f.positions] = np.nan
                df[f.variable] = arr
                self._log(f, "R-NEG", f"{f.n_rows} {f.variable} reading(s) were negative (lowest {f.evidence['min']}), "
                          "which is physically impossible and cannot be corrected without guessing. They were set to missing and "
                          "marked qc_out_of_range; the gap rule then decides whether the hour can be estimated.",
                          f.n_rows, ids(f.positions))
            else:
                self._log(f, "R-MAX", f"{f.n_rows} {f.variable} reading(s) exceed the plausible maximum of {f.evidence['limit']} "
                          f"(highest {f.evidence['max']}). This source cannot confirm or refute them, so the values were kept "
                          "and marked qc_out_of_range.", f.n_rows, ids(f.positions))

        for f in self._run(tool_pm_order, "station", station_id, df):
            flags = df["qc_pm_inconsistent"].to_numpy(copy=True)
            flags[f.positions] = True
            df["qc_pm_inconsistent"] = flags
            self._log(f, "R-PM-ORDER", f"{f.n_rows} hour(s) report PM2.5 above PM10, which is physically inconsistent. "
                      "It is unclear which value is wrong, so both were kept and the rows marked qc_pm_inconsistent.",
                      f.n_rows, ids(f.positions))

        for f in self._run(tool_gaps, "station", station_id, df):
            self._decide_gaps(f, df, ids)

        for f in self._run(tool_spikes, "station", station_id, df):
            flags = df["qc_spike"].to_numpy(copy=True)
            flags[f.positions] = True
            df["qc_spike"] = flags
            e = f.evidence
            self._log(f, "R-SPIKE", f"{f.n_rows} {f.variable} reading(s) sit more than {C.SPIKE_K:g} robust standard deviations above "
                      f"the {C.SPIKE_WINDOW}-hour rolling median (threshold {e['threshold']:.1f}, highest value {e['max']}). "
                      "Real pollution episodes look like this, so nothing was changed; rows are marked qc_spike for review.",
                      f.n_rows, ids(f.positions))
        return df

    def _decide_gaps(self, f: Finding, df: pd.DataFrame, ids) -> None:
        runs: pd.DataFrame = f.evidence["runs"].copy()
        runs["rule"] = np.select(
            [runs["edge"], runs["length"] > C.MAX_FILL_HOURS, runs["jump"] > runs["allowed"]],
            ["R-GAP-EDGE", "R-GAP-LONG", "R-GAP-JUMP"], default="R-GAP-FILL")
        interp = df[f.variable].astype(float).interpolate(method="linear", limit_area="inside").to_numpy()
        for rule, grp in runs.groupby("rule"):
            positions = np.concatenate([np.arange(s, e) for s, e in zip(grp["start"], grp["end"])])
            n_runs, hours, longest = len(grp), int(grp["length"].sum()), int(grp["length"].max())
            if rule == "R-GAP-FILL":
                arr = df[f.variable].to_numpy(dtype=float, copy=True)
                arr[positions] = interp[positions]
                df[f.variable] = arr
                flags = df["was_imputed"].to_numpy(copy=True)
                flags[positions] = True
                df["was_imputed"] = flags
                text = (f"{n_runs} interior gap(s) totalling {hours} h in {f.variable} (longest {longest} h, limit {C.MAX_FILL_HOURS} h). "
                        f"The values on either side differ by at most {grp['jump'].max():.1f}, within the {grp['allowed'].max():.1f} "
                        f"allowed by the series' own {C.SMOOTH_QUANTILE:.0%} hourly change. Linearly interpolated and marked was_imputed.")
            elif rule == "R-GAP-EDGE":
                text = (f"{n_runs} gap(s) totalling {hours} h in {f.variable} touch the start or end of the record, so there is nothing "
                        "to interpolate between. The hours stay missing (not zero).")
            elif rule == "R-GAP-LONG":
                text = (f"{n_runs} gap(s) in {f.variable} are longer than the {C.MAX_FILL_HOURS} h fill limit (longest {longest} h, "
                        f"{hours} h in total). Filling would hide real uncertainty, so the hours stay missing and are flagged.")
            else:
                text = (f"{n_runs} short gap(s) totalling {hours} h in {f.variable} sit between values that differ by up to "
                        f"{grp['jump'].max():.1f}, more than the {grp['allowed'].min():.1f} the series normally allows over that time. "
                        "Interpolating would invent a trend, so the hours stay missing and are flagged.")
            self._log(f, rule, text, len(positions), ids(positions),
                      {"runs": n_runs, "hours": hours, "longest_h": longest, "variable": f.variable})

    def review_dataset(self, frames: dict[str, pd.DataFrame], stations: pd.DataFrame) -> pd.DataFrame:
        stations = stations.sort_values("station_id").reset_index(drop=True)
        stations["is_primary_series"] = ~stations.duplicated("series_group_id", keep="first")
        for f in self._run(tool_identical_series, "dataset", "ALL", stations):
            e = f.evidence
            primary = [m for m in e["members"] if bool(stations.loc[stations["station_id"] == m, "is_primary_series"].iloc[0])][0]
            dupes = [m for m in e["members"] if m != primary]
            self._log(f, "R-IDENTICAL-SERIES",
                      f"{len(e['members'])} stations ({', '.join(e['members'])}) report byte-identical series for all variables and sit in "
                      f"model grid cell {e['cell']}, so they are one measurement counted {len(e['members'])} times. All rows were kept; "
                      f"{primary} is the primary series and {', '.join(dupes)} {'is' if len(dupes) == 1 else 'are'} excluded from gold tables to avoid double counting.",
                      e["n_rows"] * len(dupes), [])
        for f in self._run(tool_stale_sources, "dataset", "ALL", frames):
            e = f.evidence
            ts = frames[f.station_id]["ts_utc"]
            when = "never reported a valid PM2.5 value" if e["last_valid"] is None else \
                f"last reported PM2.5 at {e['last_valid']}, {e['lag_hours']:.0f} h before the newest source"
            self._log(f, "R-STALE", f"{f.station_id} {when} (grace {C.STALE_GRACE_HOURS} h). Its trailing hours stay missing and the "
                      "dashboard must show this source as stale.", f.n_rows, make_row_ids(f.station_id, ts.iloc[f.positions]))
        for f in self._run(tool_distribution_shift, "dataset", "ALL", frames):
            e = f.evidence
            self._log(f, "R-DRIFT", f"{f.variable} at {f.station_id} changed between the first and second half of the record "
                      f"(PSI {e['psi']:.2f} > {C.PSI_THRESHOLD}; mean {e['mean_first']:.1f} to {e['mean_second']:.1f}). Data were not altered; "
                      "models trained on the first half may not transfer, so the shift is flagged.", 0, [])
        return stations

    def decisions_frame(self) -> pd.DataFrame:
        cols = ["decision_id", "run_id", "decided_at_utc", "scope", "station_id", "tool", "rule_id", "action", "variable",
                "rows_affected", "policy", "explanation", "evidence", "example_row_ids"]
        return pd.DataFrame(self.decisions, columns=cols)

    def decision_rows_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.decision_rows, columns=["decision_id", "row_id"])

    def tool_runs_frame(self) -> pd.DataFrame:
        if not self.tool_runs:
            return pd.DataFrame(columns=["tool", "scope", "checks_run", "findings", "rows_flagged"])
        return (pd.DataFrame(self.tool_runs).groupby(["tool", "scope"], as_index=False)
                .agg(checks_run=("station_id", "size"), findings=("findings", "sum"), rows_flagged=("rows_flagged", "sum")))

    def quarantine_frame(self) -> pd.DataFrame:
        if not self.quarantined:
            return pd.DataFrame(columns=["station_id", "rule_id", "raw_record"])
        return pd.concat(self.quarantined, ignore_index=True)

    def report_markdown(self) -> str:
        d, t = self.decisions_frame(), self.tool_runs_frame()
        lines = [f"# Data Quality Agent report: {self.run_id}", "", "## Tools run", "",
                 "| tool | scope | checks run | findings | rows flagged |", "|---|---|---|---|---|"]
        lines += [f"| {r.tool} | {r.scope} | {r.checks_run} | {r.findings} | {r.rows_flagged} |" for r in t.itertuples()]
        counts = d["action"].value_counts().to_dict() if len(d) else {}
        lines += ["", "## Decisions", "", ", ".join(f"{k}: {v}" for k, v in counts.items()) or "No problems found.", ""]
        for action in (FIX, FILL, FLAG, QUARANTINE):
            sub = d[d["action"] == action]
            if sub.empty:
                continue
            lines += [f"### {action} ({len(sub)})", ""]
            for r in sub.head(60).itertuples():
                lines.append(f"- `{r.decision_id}` [{r.rule_id}] {r.station_id} / {r.variable}, {r.rows_affected} rows: {r.explanation}")
            if len(sub) > 60:
                lines.append(f"- ... and {len(sub) - 60} more (see quality_log.csv)")
            lines.append("")
        return "\n".join(lines)
