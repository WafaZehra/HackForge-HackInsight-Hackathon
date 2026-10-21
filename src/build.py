from __future__ import annotations

import json
import importlib
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import acquire, clean, process, validate
from . import config as C


def _promote(staged: Path, final: Path) -> None:
    old = final.with_name(final.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        final.rename(old)
    staged.rename(final)
    if old.exists():
        shutil.rmtree(old)


def _log_run(record: dict) -> None:
    C.LOGS.mkdir(parents=True, exist_ok=True)
    with open(C.RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def main() -> int:
    started = datetime.now(timezone.utc)
    run_id = started.strftime("run_%Y%m%dT%H%M%SZ")
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    def step(name: str, t_start: float) -> float:
        now = time.perf_counter()
        timings[name] = round(now - t_start, 2)
        return now

    try:
        if C.STAGING.exists():
            shutil.rmtree(C.STAGING)
        manifest = acquire.ensure_raw()
        t = step("raw", t0)
        result = clean.clean_all(manifest, run_id)
        validate.validate_cleaned(result.hourly)
        clean.write_cleaned(result, C.STAGING / "cleaned")
        t = step("cleaned", t)
        db = C.STAGING / "processed" / C.WAREHOUSE_NAME
        db.parent.mkdir(parents=True, exist_ok=True)
        process.build_warehouse(db, C.STAGING / "cleaned", manifest)
        t = step("processed", t)
        checks = validate.validate_warehouse(db, manifest)
        step("validate", t)
        try:
            duckdb = importlib.import_module("duckdb")
        except ModuleNotFoundError as exc:
            if exc.name == "duckdb":
                raise RuntimeError("DuckDB is required; install it with `pip install duckdb`.") from exc
            raise
        con = duckdb.connect(str(db), read_only=True)
        fact_rows = con.execute("SELECT count(*) FROM fact_aq_hourly").fetchone()[0]
        con.close()
        finished = datetime.now(timezone.utc)
        record = {"run_id": run_id, "started_utc": started.isoformat(timespec="seconds"),
                  "finished_utc": finished.isoformat(timespec="seconds"), "status": "success",
                  "fact_rows": int(fact_rows), "fact_content_hash": process.content_hash(db)}
        process.record_run(db, record)
        _promote(C.STAGING / "cleaned", C.CLEANED)
        _promote(C.STAGING / "processed", C.PROCESSED)
        shutil.rmtree(C.STAGING, ignore_errors=True)
        _log_run({**record, "timings_s": timings})
    except Exception as exc:
        finished = datetime.now(timezone.utc)
        _log_run({"run_id": run_id, "started_utc": started.isoformat(timespec="seconds"),
                  "finished_utc": finished.isoformat(timespec="seconds"), "status": "failed",
                  "error": f"{type(exc).__name__}: {exc}", "timings_s": timings})
        print(f"\n[FAILED] {type(exc).__name__}: {exc}\nPrevious cleaned/processed layers were kept.", file=sys.stderr)
        if not isinstance(exc, (validate.ValidationError, acquire.RawIntegrityError, FileNotFoundError)):
            traceback.print_exc()
        shutil.rmtree(C.STAGING, ignore_errors=True)
        return 1
    _print_summary(result, checks, record, timings)
    return 0


def _print_summary(result: clean.CleanResult, checks, record: dict, timings: dict) -> None:
    h, s, d = result.hourly, result.stations, result.decisions
    print("\n=== make data: OK ===")
    print(f"run_id            {record['run_id']}")
    print(f"stations          {len(s)} supplied, {int(s['is_primary_series'].sum())} unique series (primary)")
    print(f"hourly rows       {len(h):,}   {h['ts_utc'].min():%Y-%m-%d %H:%M} -> {h['ts_utc'].max():%Y-%m-%d %H:%M} UTC")
    print(f"flags             gap_rows={int(h['is_gap_row'].sum()):,} imputed={int(h['was_imputed'].sum()):,} "
          f"out_of_range={int(h['qc_out_of_range'].sum()):,} pm25>pm10={int(h['qc_pm_inconsistent'].sum()):,} "
          f"spikes={int(h['qc_spike'].sum()):,}")
    t = result.tool_runs
    print(f"DQ agent          {int(t['checks_run'].sum())} tool runs across {len(t)} tools, {int(t['findings'].sum())} findings")
    print(f"decisions logged  {len(d)}  ({', '.join(f'{k}={v}' for k, v in d['action'].value_counts().items()) or 'none'})")
    print(f"quarantined rows  {len(result.quarantine):,}")
    print(f"content hash      {record['fact_content_hash']}")
    print("timings (s)       " + "  ".join(f"{k}={v}" for k, v in timings.items()))
    print("gate 2 checks:")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


if __name__ == "__main__":
    sys.exit(main())
