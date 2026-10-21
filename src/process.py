from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from . import config as C


def _naive_utc(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def build_warehouse(db_path: Path, cleaned_dir: Path, manifest: pd.DataFrame) -> None:
    if db_path.exists():
        db_path.unlink()
    hourly = _naive_utc(pd.read_parquet(cleaned_dir / "aq_hourly_cleaned.parquet"), ["ts_utc"])
    stations = _naive_utc(pd.read_parquet(cleaned_dir / "stations.parquet"), ["first_ts_utc", "last_ts_utc"])
    qlog = pd.read_csv(cleaned_dir / "quality_log.csv")
    drows = pd.read_csv(cleaned_dir / "dq_decision_rows.csv")
    truns = pd.read_csv(cleaned_dir / "dq_tool_runs.csv")

    who25, who10 = C.WHO_2021_24H["pm25"], C.WHO_2021_24H["pm10"]
    min_h, off = C.MIN_HOURS_PER_DAY, C.PKT_OFFSET_HOURS

    con = duckdb.connect(str(db_path))
    try:
        con.register("hourly_df", hourly)
        con.register("stations_df", stations)
        con.register("manifest_df", manifest)
        con.register("qlog_df", qlog)
        con.register("drows_df", drows)
        con.register("truns_df", truns)

        con.execute("CREATE TABLE dim_station AS SELECT * FROM stations_df ORDER BY station_id")
        con.execute("""
            CREATE TABLE fact_aq_hourly AS
            SELECT station_id || ':' || strftime(ts_utc, '%Y%m%d%H') AS row_id, *
            FROM hourly_df ORDER BY station_id, ts_utc
        """)
        con.execute(f"""
            CREATE TABLE gold_station_daily AS
            WITH h AS (
                SELECT f.station_id, d.city, CAST(f.ts_utc + INTERVAL {off} HOUR AS DATE) AS date_pkt, f.pm25, f.pm10, f.no2, f.so2, f.o3, f.co
                FROM fact_aq_hourly f JOIN dim_station d ON f.station_id = d.station_id
                WHERE d.is_primary_series
            ), agg AS (
                SELECT station_id, city, date_pkt,
                       count(pm25) AS n_hours_pm25, count(pm10) AS n_hours_pm10, max(pm25) AS pm25_max_1h,
                       CASE WHEN count(pm25) >= {min_h} THEN avg(pm25) END AS pm25_mean_24h,
                       CASE WHEN count(pm10) >= {min_h} THEN avg(pm10) END AS pm10_mean_24h,
                       CASE WHEN count(no2) >= {min_h} THEN avg(no2) END AS no2_mean_24h,
                       CASE WHEN count(so2) >= {min_h} THEN avg(so2) END AS so2_mean_24h,
                       CASE WHEN count(o3) >= {min_h} THEN avg(o3) END AS o3_mean_24h,
                       CASE WHEN count(co) >= {min_h} THEN avg(co) END AS co_mean_24h
                FROM h GROUP BY station_id, city, date_pkt
            )
            SELECT station_id || ':' || strftime(date_pkt, '%Y%m%d') AS row_id, *,
                   pm25_mean_24h > {who25} AS exceeds_who_pm25_24h,
                   pm10_mean_24h > {who10} AS exceeds_who_pm10_24h
            FROM agg ORDER BY station_id, date_pkt
        """)
        con.execute("""
            CREATE TABLE gold_city_hourly AS
            SELECT lower(d.city) || ':' || strftime(f.ts_utc, '%Y%m%d%H') AS row_id, d.city, f.ts_utc,
                   avg(f.pm25) AS pm25_mean, min(f.pm25) AS pm25_min, max(f.pm25) AS pm25_max,
                   avg(f.pm10) AS pm10_mean, avg(f.no2) AS no2_mean, avg(f.o3) AS o3_mean,
                   avg(f.so2) AS so2_mean, avg(f.co) AS co_mean,
                   count(f.pm25) AS n_stations_pm25, count(*) AS n_stations
            FROM fact_aq_hourly f JOIN dim_station d ON f.station_id = d.station_id
            WHERE d.is_primary_series
            GROUP BY d.city, f.ts_utc ORDER BY d.city, f.ts_utc
        """)
        con.execute(f"""
            CREATE TABLE gold_city_daily AS
            WITH agg AS (
                SELECT city, date_pkt,
                       avg(pm25_mean_24h) AS pm25_mean_24h, avg(pm10_mean_24h) AS pm10_mean_24h,
                       count(pm25_mean_24h) AS n_stations_complete, count(*) AS n_stations
                FROM gold_station_daily GROUP BY city, date_pkt
            )
            SELECT lower(city) || ':' || strftime(date_pkt, '%Y%m%d') AS row_id, *,
                   pm25_mean_24h > {who25} AS exceeds_who_pm25_24h,
                   pm10_mean_24h > {who10} AS exceeds_who_pm10_24h
            FROM agg ORDER BY city, date_pkt
        """)
        con.execute("""
            CREATE TABLE meta_source_freshness AS
            SELECT station_id, max(ts_utc) FILTER (WHERE pm25 IS NOT NULL) AS last_valid_ts_utc,
                   count(*) AS n_rows, count(*) FILTER (WHERE pm25 IS NULL) AS n_missing_pm25
            FROM fact_aq_hourly GROUP BY station_id ORDER BY station_id
        """)
        con.execute("CREATE TABLE meta_manifest AS SELECT * FROM manifest_df")
        con.execute("CREATE TABLE meta_quality_log AS SELECT * FROM qlog_df")
        con.execute("CREATE TABLE meta_dq_decision_rows AS SELECT * FROM drows_df")
        con.execute("CREATE TABLE meta_dq_tool_runs AS SELECT * FROM truns_df")
        con.execute("""
            CREATE TABLE meta_pipeline_run (
                run_id VARCHAR, started_utc VARCHAR, finished_utc VARCHAR, status VARCHAR,
                fact_rows BIGINT, fact_content_hash VARCHAR
            )
        """)
    finally:
        con.close()


def record_run(db_path: Path, run: dict) -> None:
    con = duckdb.connect(str(db_path))
    try:
        con.execute("INSERT INTO meta_pipeline_run VALUES (?, ?, ?, ?, ?, ?)",
                    [run["run_id"], run["started_utc"], run["finished_utc"], run["status"],
                     run["fact_rows"], run["fact_content_hash"]])
    finally:
        con.close()


def content_hash(db_path: Path) -> str:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            "SELECT lpad(to_hex(sum(hash(row_id, pm10, pm25, co, no2, o3, so2) % 4294967296)::BIGINT), 12, '0') "
            "FROM fact_aq_hourly").fetchone()[0]
    finally:
        con.close()
