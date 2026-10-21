"""Quality gates. A failed gate raises ValidationError; build.py then keeps the last good layers."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pandera.pandas as pa

from . import config as C
from .acquire import verify_raw
from .common import ValidationError


def validate_cleaned(hourly: pd.DataFrame) -> None:
    """Gate 1 (schema/contract) on the cleaned layer, using pandera."""
    schema = pa.DataFrameSchema(
        {
            "station_id": pa.Column(str, nullable=False),
            "ts_utc": pa.Column(nullable=False),
            **{v: pa.Column(float, checks=pa.Check.ge(0), nullable=True) for v in C.POLLUTANTS},
            **{f: pa.Column(bool, nullable=False) for f in ("is_gap_row", "was_imputed", "qc_out_of_range", "qc_pm_inconsistent")},
        },
        unique=["station_id", "ts_utc"],
        strict=True,
    )
    try:
        schema.validate(hourly, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValidationError(f"cleaned layer failed schema gate:\n{exc.failure_cases.head(10)}") from exc


def validate_warehouse(db_path: Path, manifest: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """Gate 2 (completeness/consistency/volume) on the warehouse. Returns the checks; raises if any fail."""
    con = duckdb.connect(str(db_path), read_only=True)
    q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    try:
        n_fact = q("SELECT count(*) FROM fact_aq_hourly")
        n_st = q("SELECT count(DISTINCT station_id) FROM fact_aq_hourly")
        n_ts = q("SELECT count(DISTINCT ts_utc) FROM fact_aq_hourly")
        check("fact grid complete (stations x hours)", n_fact == n_st * n_ts, f"{n_fact:,} = {n_st} x {n_ts:,}")
        check("fact row_id unique", q("SELECT count(DISTINCT row_id) FROM fact_aq_hourly") == n_fact, "")
        check("one fact table row per manifest file", n_st == len(manifest), f"{n_st} stations / {len(manifest)} files")
        check("volume >= 50,000 rows", n_fact >= C.MIN_TOTAL_ROWS, f"{n_fact:,} rows")
        check("each station >= 2 years hourly", n_ts >= C.MIN_HOURS_PER_STATION, f"{n_ts:,} hourly steps")
        neg = q("SELECT count(*) FROM fact_aq_hourly WHERE pm10<0 OR pm25<0 OR co<0 OR no2<0 OR o3<0 OR so2<0")
        check("no negative concentrations", neg == 0, f"{neg} found")
        check("exactly one primary series per group",
              q("SELECT count(*) FROM (SELECT series_group_id FROM dim_station GROUP BY 1 "
                "HAVING sum(CASE WHEN is_primary_series THEN 1 ELSE 0 END) <> 1)") == 0, "")
        for t in ("gold_station_daily", "gold_city_hourly", "gold_city_daily"):
            n, nd = q(f"SELECT count(*) FROM {t}"), q(f"SELECT count(DISTINCT row_id) FROM {t}")
            check(f"{t}: non-empty, row_id unique", n > 0 and n == nd, f"{n:,} rows")
        cities = q("SELECT count(DISTINCT city) FROM gold_city_hourly")
        check("gold covers every city", cities == q("SELECT count(DISTINCT city) FROM dim_station"), f"{cities} cities")
        check("gold excludes duplicate series",
              q("SELECT count(*) FROM gold_station_daily g JOIN dim_station d USING (station_id) "
                "WHERE NOT d.is_primary_series") == 0, "")
        missing, bad = verify_raw(manifest)
        check("raw files match manifest SHA-256", not missing and not bad, f"missing={len(missing)} changed={len(bad)}")
    finally:
        con.close()

    failed = [c for c in checks if not c[1]]
    if failed:
        raise ValidationError("warehouse gate failed: " + "; ".join(f"{n} ({d})" for n, _, d in failed))
    return checks
