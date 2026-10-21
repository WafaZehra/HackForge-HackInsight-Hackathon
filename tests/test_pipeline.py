"""Unit tests. Inputs are tiny hand-built Open-Meteo-style CSVs: SYNTHETIC STRESS TESTS of the cleaning
rules only. They are never used for training or evaluation (event rule 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import acquire, clean, validate
from src import config as C
from src.common import DataContractError, parse_open_meteo_text, sha256_bytes

HEADER = "latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n31.5,74.3,217.0,0,GMT,GMT\n\n"


def make_csv(rows: list[str], unit: str = "μg/m³") -> str:
    cols = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone", "sulphur_dioxide"]
    head = "time," + ",".join(f"{c} ({unit})" for c in cols)
    return HEADER + head + "\n" + "\n".join(rows) + "\n"


def row(hour: int, v: float = 10.0, day: str = "2025-01-01") -> str:
    return f"{day}T{hour:02d}:00," + ",".join([str(v * 2), str(v), "300", "20", "50", "10"])


def clean_one(text: str, n_hours: int = 12):
    log = clean.DecisionLog("t")
    meta, raw = parse_open_meteo_text(text)
    df = clean.canonicalize_columns(raw, "s1", log)
    df = clean.parse_time_and_dedupe(df, int(meta["utc_offset_seconds"]), "s1", log)
    grid = pd.date_range("2025-01-01", periods=n_hours, freq="1h", tz="UTC").astype("datetime64[us, UTC]")
    return clean.regularize(df, grid, "s1", log), log


def test_parse_splits_metadata_and_hourly():
    meta, hourly = parse_open_meteo_text(make_csv([row(0)]))
    assert meta["latitude"] == 31.5 and list(hourly.columns)[0] == "time" and len(hourly) == 1


def test_columns_are_renamed_to_canonical():
    out, _ = clean_one(make_csv([row(h) for h in range(12)]))
    assert {"pm10", "pm25", "co", "no2", "o3", "so2"} <= set(out.columns)


def test_micro_sign_variant_is_accepted():
    out, _ = clean_one(make_csv([row(h) for h in range(12)], unit="µg/m³"))  # U+00B5
    assert out["pm25"].notna().all()


def test_mg_unit_is_converted_and_logged_as_fix():
    out, log = clean_one(make_csv([row(h, 0.01) for h in range(12)], unit="mg/m³"))
    assert out["pm25"].iloc[0] == pytest.approx(10.0)
    assert any(d["rule_id"] == "R-UNIT" and d["action"] == "FIX" for d in log.decisions)


def test_unknown_unit_is_a_contract_error():
    with pytest.raises(DataContractError):
        clean_one(make_csv([row(0)], unit="ppm"))


def test_short_gap_filled_and_marked():
    rows = [row(h, 10.0 + h) for h in range(12) if h not in (4, 5)]
    out, log = clean_one(make_csv(rows))
    assert out.loc[[4, 5], "pm25"].notna().all() and out.loc[[4, 5], "was_imputed"].all()
    assert out.loc[4, "pm25"] == pytest.approx(14.0)  # linear between 13 (h3) and 16 (h6)
    assert any(d["action"] == "FILL" for d in log.decisions)


def test_long_gap_is_flagged_not_filled():
    rows = [row(h) for h in range(12) if h not in (3, 4, 5, 6, 7)]  # 5-hour gap > MAX_FILL_HOURS
    out, log = clean_one(make_csv(rows))
    assert out.loc[3:7, "pm25"].isna().all() and not out["was_imputed"].any()
    assert out.loc[3:7, "is_gap_row"].all()
    assert any(d["rule_id"] == "R-LONG-GAP" and d["action"] == "FLAG" for d in log.decisions)


def test_edge_gaps_are_never_extrapolated():
    out, _ = clean_one(make_csv([row(h) for h in range(2, 12)]))  # hours 0-1 missing at the start
    assert out.loc[0:1, "pm25"].isna().all()


def test_negative_becomes_missing_and_flagged():
    rows = [row(h) for h in range(12)]
    rows[6] = "2025-01-01T06:00,20,-5,300,20,50,10"
    out, log = clean_one(make_csv(rows))
    # the impossible value is flagged, nulled, then (single interior hour) re-estimated and marked as imputed
    assert out.loc[6, "qc_out_of_range"] and out.loc[6, "was_imputed"] and out.loc[6, "pm25"] >= 0
    assert any(d["rule_id"] == "R-NEG" for d in log.decisions)


def test_implausibly_high_value_is_kept_and_flagged():
    rows = [row(h) for h in range(12)]
    rows[2] = "2025-01-01T02:00,5000,3000,300,20,50,10"
    out, _ = clean_one(make_csv(rows))
    assert out.loc[2, "pm25"] == 3000 and out.loc[2, "qc_out_of_range"]


def test_duplicate_timestamp_keeps_last_and_quarantines_earlier():
    rows = [row(h) for h in range(12)] + ["2025-01-01T03:00,1,99,1,1,1,1"]
    out, log = clean_one(make_csv(rows))
    assert out.loc[3, "pm25"] == 99 and out["ts_utc"].is_unique
    assert len(log.quarantine_frame()) == 1


def test_unparseable_time_is_quarantined_not_dropped_silently():
    rows = [row(h) for h in range(12)] + ["not-a-time,1,1,1,1,1,1"]
    _, log = clean_one(make_csv(rows))
    assert len(log.quarantine_frame()) == 1 and any(d["action"] == "QUARANTINE" for d in log.decisions)


def test_pm25_above_pm10_is_flagged():
    rows = [row(h) for h in range(12)]
    rows[1] = "2025-01-01T01:00,5,9,300,20,50,10"
    out, _ = clean_one(make_csv(rows))
    assert out.loc[1, "qc_pm_inconsistent"]


def test_identical_series_get_one_primary():
    log = clean.DecisionLog("t")
    st = pd.DataFrame({"station_id": ["b", "a", "c"], "series_group_id": ["g1", "g1", "g2"], "n_rows": [10, 10, 10]})
    out = clean.flag_identical_series(st, log).set_index("station_id")
    assert out.loc["a", "is_primary_series"] and not out.loc["b", "is_primary_series"] and out.loc["c", "is_primary_series"]
    assert any(d["rule_id"] == "R-IDENTICAL-SERIES" for d in log.decisions)


def test_station_identity_fixes_typos():
    assert acquire.station_identity("Peshawer", "Peshawer_Saddar.csv") == ("Peshawar", "Saddar", "peshawar__saddar")
    assert acquire.station_identity("Hunza", "Humza_Gulmit.csv")[1] == "Gulmit"
    assert acquire.station_identity("Lahore", "lahore_shahdara.csv")[1] == "Shahdara"


def test_pandera_gate_rejects_duplicate_keys_and_negatives():
    out, _ = clean_one(make_csv([row(h) for h in range(12)]))
    validate.validate_cleaned(out)  # clean frame passes
    bad = pd.concat([out, out.iloc[[0]]], ignore_index=True)
    with pytest.raises(validate.ValidationError):
        validate.validate_cleaned(bad)
    neg = out.copy()
    neg.loc[0, "pm25"] = -1.0
    with pytest.raises(validate.ValidationError):
        validate.validate_cleaned(neg)


def test_verify_raw_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RAW", tmp_path)
    (tmp_path / "X").mkdir()
    f = tmp_path / "X" / "a.csv"
    content = make_csv([row(0)]).encode()
    f.write_bytes(content)
    manifest = pd.DataFrame({"file": ["X/a.csv", "X/missing.csv"], "sha256": [sha256_bytes(content)] * 2})
    missing, bad = acquire.verify_raw(manifest)
    assert missing == ["X/missing.csv"] and bad == []
    f.write_bytes(content + b"tampered")
    assert acquire.verify_raw(manifest)[1] == ["X/a.csv"]


def test_ingest_is_byte_identical_and_immutable(tmp_path, monkeypatch):
    src = tmp_path / "in" / "Lahore"
    src.mkdir(parents=True)
    content = make_csv([row(h) for h in range(3)]).encode()
    (src / "lahore_Gulberg.csv").write_bytes(content)
    monkeypatch.setattr(C, "RAW", tmp_path / "raw")
    monkeypatch.setattr(C, "MANIFEST", tmp_path / "manifest.csv")
    m = acquire.ingest(tmp_path / "in")
    assert (tmp_path / "raw" / "Lahore" / "lahore_Gulberg.csv").read_bytes() == content
    assert m.loc[0, "sha256"] == sha256_bytes(content) and m.loc[0, "rows"] == 3
    (src / "lahore_Gulberg.csv").write_bytes(content + b"x")  # different content, same path
    with pytest.raises(acquire.RawIntegrityError):
        acquire.ingest(tmp_path / "in")


def test_gold_daily_uses_pkt_day_and_completeness_rule(tmp_path):
    """End-to-end on a 2-day synthetic station: PKT day = UTC+5, and a day with < 18 valid hours gets no mean."""
    import duckdb
    from src import process
    hours = pd.date_range("2025-01-01 00:00", periods=48, freq="1h", tz="UTC").astype("datetime64[us, UTC]")
    df = pd.DataFrame({"station_id": "karachi__s1", "ts_utc": hours})
    for v in C.POLLUTANTS:
        df[v] = 10.0
    df["pm25"] = np.where(np.arange(48) >= 30, np.nan, 20.0)  # last 18 hours missing
    for f in clean.FLAG_COLS:
        df[f] = False
    st = pd.DataFrame([{"station_id": "karachi__s1", "city": "Karachi", "station": "S1", "latitude": 24.9, "longitude": 67.1,
                        "elevation_m": 5.0, "grid_cell_id": "x", "series_group_id": "g", "n_rows": 48,
                        "first_ts_utc": hours.min(), "last_ts_utc": hours.max(), "source_file": "f", "sha256": "h",
                        "is_primary_series": True}])
    cdir = tmp_path / "cleaned"
    cdir.mkdir()
    df.to_parquet(cdir / "aq_hourly_cleaned.parquet", index=False)
    st.to_parquet(cdir / "stations.parquet", index=False)
    pd.DataFrame(columns=["run_id"]).to_csv(cdir / "quality_log.csv", index=False)
    db = tmp_path / "w.duckdb"
    process.build_warehouse(db, cdir, pd.DataFrame({"file": ["f"]}))
    out = duckdb.connect(str(db), read_only=True).sql(
        "select date_pkt, n_hours_pm25, pm25_mean_24h, exceeds_who_pm25_24h from gold_station_daily order by 1").df()
    # 48 UTC hours from 00:00 touch three PKT days (05:00 Jan 1 -> 04:00 Jan 3)
    assert out["date_pkt"].astype(str).tolist() == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert out.loc[0, "pm25_mean_24h"] == 20.0 and bool(out.loc[0, "exceeds_who_pm25_24h"])  # 19 valid hours, kept
    assert np.isnan(out.loc[1, "pm25_mean_24h"])  # only 11 valid hours -> no daily mean
