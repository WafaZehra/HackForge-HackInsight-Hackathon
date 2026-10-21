from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import acquire, clean, validate
from src import config as C
from src.common import DataContractError, parse_open_meteo_text, sha256_bytes
from src.dq_agent import FLAG_COLUMNS, DataQualityAgent

HEADER = "latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n31.5,74.3,217.0,0,GMT,GMT\n\n"


def synthetic_csv(rows, unit="μg/m³"):
    cols = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone", "sulphur_dioxide"]
    head = "time," + ",".join(f"{c} ({unit})" for c in cols)
    return HEADER + head + "\n" + "\n".join(rows) + "\n"


def row(hour, v=10.0, day="2025-01-01"):
    return f"{day}T{hour:02d}:00," + ",".join([str(v * 2), str(v), "300", "20", "50", "10"])


def run_agent(text, n_hours=12):
    agent = DataQualityAgent("t")
    meta, raw = parse_open_meteo_text(text)
    df = agent.review_raw("s1", meta, raw)
    grid = pd.date_range("2025-01-01", periods=n_hours, freq="1h", tz="UTC").astype("datetime64[us, UTC]")
    return agent.review_station("s1", clean.put_on_grid(df, grid, "s1")), agent


def rules(agent):
    return [d["rule_id"] for d in agent.decisions]


def test_parse_splits_metadata_and_hourly():
    meta, hourly = parse_open_meteo_text(synthetic_csv([row(0)]))
    assert meta["latitude"] == 31.5 and hourly.columns[0] == "time" and len(hourly) == 1


def test_columns_renamed_and_clean_input_needs_no_decisions():
    out, agent = run_agent(synthetic_csv([row(h) for h in range(12)]))
    assert {"pm10", "pm25", "co", "no2", "o3", "so2"} <= set(out.columns)
    assert agent.decisions == []
    assert agent.tool_runs_frame()["checks_run"].sum() > 0


def test_micro_sign_variant_accepted():
    out, _ = run_agent(synthetic_csv([row(h) for h in range(12)], unit="µg/m³"))
    assert out["pm25"].notna().all()


def test_unit_conversion_is_fix_with_explanation():
    out, agent = run_agent(synthetic_csv([row(h, 0.01) for h in range(12)], unit="mg/m³"))
    assert out["pm25"].iloc[0] == pytest.approx(10.0)
    d = agent.decisions[0]
    assert d["action"] == "FIX" and d["rule_id"] == "R-UNIT" and "1000" in d["explanation"]


def test_unknown_unit_stops_the_run():
    with pytest.raises(DataContractError):
        run_agent(synthetic_csv([row(0)], unit="ppm"))


def test_short_smooth_gap_is_filled_marked_and_explained():
    rows = [row(h, 10.0 + h) for h in range(12) if h not in (4, 5)]
    out, agent = run_agent(synthetic_csv(rows))
    assert out.loc[[4, 5], "was_imputed"].all() and out.loc[4, "pm25"] == pytest.approx(14.0)
    fills = [d for d in agent.decisions if d["action"] == "FILL"]
    assert fills and all(len(d["explanation"]) > 40 for d in fills)
    assert fills[0]["example_row_ids"].startswith("s1:2025010104")


def test_long_gap_is_flagged_not_filled():
    rows = [row(h) for h in range(12) if h not in (3, 4, 5, 6, 7)]
    out, agent = run_agent(synthetic_csv(rows))
    assert out.loc[3:7, "pm25"].isna().all() and not out["was_imputed"].any()
    assert "R-GAP-LONG" in rules(agent) and all(d["action"] == "FLAG" for d in agent.decisions)


def test_short_gap_with_big_jump_is_flagged_not_filled():
    stamps = pd.date_range("2025-01-01", periods=200, freq="1h")
    rows = []
    for i, t in enumerate(stamps):
        if i == 100:
            continue
        v = 500.0 if i == 101 else 10.0 + 0.1 * (i % 5)
        rows.append(f"{t:%Y-%m-%dT%H:%M}," + ",".join([str(v * 2), str(v), "300", "20", "50", "10"]))
    out, agent = run_agent(synthetic_csv(rows), n_hours=200)
    assert np.isnan(out.loc[100, "pm25"]) and "R-GAP-JUMP" in rules(agent)


def test_edge_gap_never_extrapolated():
    out, agent = run_agent(synthetic_csv([row(h) for h in range(2, 12)]))
    assert out.loc[0:1, "pm25"].isna().all() and "R-GAP-EDGE" in rules(agent)


def test_negative_is_flagged_nulled_then_gap_rule_decides():
    rows = [row(h) for h in range(12)]
    rows[6] = "2025-01-01T06:00,20,-5,300,20,50,10"
    out, agent = run_agent(synthetic_csv(rows))
    assert out.loc[6, "qc_out_of_range"] and out.loc[6, "was_imputed"]
    assert rules(agent)[0] == "R-NEG" and "R-GAP-FILL" in rules(agent)


def test_implausibly_high_value_kept_and_flagged():
    rows = [row(h) for h in range(12)]
    rows[2] = "2025-01-01T02:00,5000,3000,300,20,50,10"
    out, agent = run_agent(synthetic_csv(rows))
    assert out.loc[2, "pm25"] == 3000 and out.loc[2, "qc_out_of_range"] and "R-MAX" in rules(agent)


def test_duplicate_timestamp_fixed_and_earlier_copy_quarantined():
    rows = [row(h) for h in range(12)] + ["2025-01-01T03:00,1,99,1,1,1,1"]
    out, agent = run_agent(synthetic_csv(rows))
    assert out.loc[3, "pm25"] == 99 and out["ts_utc"].is_unique
    assert len(agent.quarantine_frame()) == 1 and agent.decisions[0]["action"] == "FIX"


def test_unparseable_time_quarantined():
    rows = [row(h) for h in range(12)] + ["not-a-time,1,1,1,1,1,1"]
    _, agent = run_agent(synthetic_csv(rows))
    assert len(agent.quarantine_frame()) == 1 and agent.decisions[0]["action"] == "QUARANTINE"


def test_pm25_above_pm10_flagged():
    rows = [row(h) for h in range(12)]
    rows[1] = "2025-01-01T01:00,5,9,300,20,50,10"
    out, agent = run_agent(synthetic_csv(rows))
    assert out.loc[1, "qc_pm_inconsistent"] and "R-PM-ORDER" in rules(agent)


def test_spike_is_flagged_and_value_kept():
    agent = DataQualityAgent("t")
    df = pd.DataFrame({"ts_utc": pd.date_range("2025-01-01", periods=48, freq="1h", tz="UTC").astype("datetime64[us, UTC]")})
    for v in C.POLLUTANTS:
        df[v] = 10.0 + (np.arange(48) % 3)
    df.loc[20, "pm25"] = 600.0
    out = agent.review_station("s1", clean.put_on_grid(df, pd.DatetimeIndex(df["ts_utc"]), "s1"))
    assert out.loc[20, "qc_spike"] and out.loc[20, "pm25"] == 600.0
    assert any(d["rule_id"] == "R-SPIKE" and d["action"] == "FLAG" for d in agent.decisions)


def test_identical_series_get_one_primary():
    agent = DataQualityAgent("t")
    st = pd.DataFrame({"station_id": ["b", "a", "c"], "series_group_id": ["g1", "g1", "g2"], "n_rows": [10, 10, 10],
                       "grid_cell_id": ["x", "x", "y"]})
    frames = {k: pd.DataFrame({"ts_utc": pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC"),
                               **{v: [1.0, 2.0, 3.0] for v in C.POLLUTANTS}}) for k in "abc"}
    out = agent.review_dataset(frames, st).set_index("station_id")
    assert out.loc["a", "is_primary_series"] and not out.loc["b", "is_primary_series"] and out.loc["c", "is_primary_series"]
    assert "R-IDENTICAL-SERIES" in rules(agent) and "a is the primary series" in agent.decisions[0]["explanation"]


def test_stale_source_is_flagged_with_row_ids():
    agent = DataQualityAgent("t")
    ts = pd.date_range("2025-01-01", periods=48, freq="1h", tz="UTC")
    live = pd.DataFrame({"ts_utc": ts, **{v: np.ones(48) for v in C.POLLUTANTS}})
    dead = live.copy()
    dead.loc[24:, list(C.POLLUTANTS)] = np.nan
    st = pd.DataFrame({"station_id": ["dead", "live"], "series_group_id": ["g1", "g2"], "n_rows": [48, 48], "grid_cell_id": ["x", "y"]})
    agent.review_dataset({"live": live, "dead": dead}, st)
    stale = [d for d in agent.decisions if d["rule_id"] == "R-STALE"]
    assert len(stale) == 1 and stale[0]["station_id"] == "dead" and stale[0]["rows_affected"] == 24
    assert len(agent.decision_rows_frame()) == 24


def test_distribution_shift_is_flagged():
    agent = DataQualityAgent("t")
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({"ts_utc": pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")})
    for v in C.POLLUTANTS:
        df[v] = np.r_[rng.normal(10, 2, n // 2), rng.normal(40, 2, n // 2)]
    st = pd.DataFrame({"station_id": ["s"], "series_group_id": ["g"], "n_rows": [n], "grid_cell_id": ["x"]})
    agent.review_dataset({"s": df}, st)
    assert "R-DRIFT" in rules(agent)


def test_every_decision_has_explanation_and_unique_id():
    rows = [row(h) for h in range(12) if h not in (4, 5, 6, 7, 8)] + ["2025-01-01T03:00,1,99,1,1,1,1"]
    _, agent = run_agent(synthetic_csv(rows))
    d = agent.decisions_frame()
    assert d["decision_id"].is_unique and (d["explanation"].str.len() > 40).all() and d["action"].isin(
        ["FIX", "FILL", "FLAG", "QUARANTINE"]).all()


def test_report_lists_tools_and_decisions():
    _, agent = run_agent(synthetic_csv([row(h) for h in range(12) if h != 5]))
    text = agent.report_markdown()
    assert "## Tools run" in text and "tool_gaps" in text and "FILL" in text


def test_station_identity_fixes_typos():
    assert acquire.station_identity("Peshawer", "Peshawer_Saddar.csv") == ("Peshawar", "Saddar", "peshawar__saddar")
    assert acquire.station_identity("Hunza", "Humza_Gulmit.csv")[1] == "Gulmit"
    assert acquire.station_identity("Lahore", "lahore_shahdara.csv")[1] == "Shahdara"


def test_pandera_gate_rejects_duplicate_keys_and_negatives():
    out, _ = run_agent(synthetic_csv([row(h) for h in range(12)]))
    validate.validate_cleaned(out)
    with pytest.raises(validate.ValidationError):
        validate.validate_cleaned(pd.concat([out, out.iloc[[0]]], ignore_index=True))
    neg = out.copy()
    neg.loc[0, "pm25"] = -1.0
    with pytest.raises(validate.ValidationError):
        validate.validate_cleaned(neg)


def test_verify_raw_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RAW", tmp_path)
    (tmp_path / "X").mkdir()
    f = tmp_path / "X" / "a.csv"
    content = synthetic_csv([row(0)]).encode()
    f.write_bytes(content)
    manifest = pd.DataFrame({"file": ["X/a.csv", "X/missing.csv"], "sha256": [sha256_bytes(content)] * 2})
    missing, bad = acquire.verify_raw(manifest)
    assert missing == ["X/missing.csv"] and bad == []
    f.write_bytes(content + b"tampered")
    assert acquire.verify_raw(manifest)[1] == ["X/a.csv"]


def test_ingest_is_byte_identical_and_immutable(tmp_path, monkeypatch):
    src = tmp_path / "in" / "Lahore"
    src.mkdir(parents=True)
    content = synthetic_csv([row(h) for h in range(3)]).encode()
    (src / "lahore_Gulberg.csv").write_bytes(content)
    monkeypatch.setattr(C, "RAW", tmp_path / "raw")
    monkeypatch.setattr(C, "MANIFEST", tmp_path / "manifest.csv")
    m = acquire.ingest(tmp_path / "in")
    assert (tmp_path / "raw" / "Lahore" / "lahore_Gulberg.csv").read_bytes() == content
    assert m.loc[0, "sha256"] == sha256_bytes(content) and m.loc[0, "rows"] == 3
    (src / "lahore_Gulberg.csv").write_bytes(content + b"x")
    with pytest.raises(acquire.RawIntegrityError):
        acquire.ingest(tmp_path / "in")


def test_gold_daily_uses_pkt_day_and_completeness_rule(tmp_path):
    import duckdb
    from src import process
    hours = pd.date_range("2025-01-01 00:00", periods=48, freq="1h", tz="UTC").astype("datetime64[us, UTC]")
    df = pd.DataFrame({"station_id": "karachi__s1", "ts_utc": hours})
    for v in C.POLLUTANTS:
        df[v] = 10.0
    df["pm25"] = np.where(np.arange(48) >= 30, np.nan, 20.0)
    for f in FLAG_COLUMNS:
        df[f] = False
    st = pd.DataFrame([{"station_id": "karachi__s1", "city": "Karachi", "station": "S1", "latitude": 24.9, "longitude": 67.1,
                        "elevation_m": 5.0, "grid_cell_id": "x", "series_group_id": "g", "n_rows": 48,
                        "first_ts_utc": hours.min(), "last_ts_utc": hours.max(), "source_file": "f", "sha256": "h",
                        "is_primary_series": True}])
    cdir = tmp_path / "cleaned"
    cdir.mkdir()
    df.to_parquet(cdir / "aq_hourly_cleaned.parquet", index=False)
    st.to_parquet(cdir / "stations.parquet", index=False)
    pd.DataFrame(columns=["decision_id", "explanation"]).to_csv(cdir / "quality_log.csv", index=False)
    pd.DataFrame(columns=["decision_id", "row_id"]).to_csv(cdir / "dq_decision_rows.csv", index=False)
    pd.DataFrame(columns=["tool", "scope", "checks_run", "findings", "rows_flagged"]).to_csv(cdir / "dq_tool_runs.csv", index=False)
    db = tmp_path / "w.duckdb"
    process.build_warehouse(db, cdir, pd.DataFrame({"file": ["f"]}))
    out = duckdb.connect(str(db), read_only=True).sql(
        "select date_pkt, n_hours_pm25, pm25_mean_24h, exceeds_who_pm25_24h from gold_station_daily order by 1").df()
    assert out["date_pkt"].astype(str).tolist() == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert out.loc[0, "pm25_mean_24h"] == 20.0 and bool(out.loc[0, "exceeds_who_pm25_24h"])
    assert np.isnan(out.loc[1, "pm25_mean_24h"])


def test_ingest_skips_identical_duplicate_copies(tmp_path, monkeypatch):
    content = synthetic_csv([row(h) for h in range(3)]).encode()
    for sub in ("Raw/Lahore", "Data/Raw/Lahore"):
        d = tmp_path / "in" / sub
        d.mkdir(parents=True)
        (d / "lahore_Gulberg.csv").write_bytes(content)
    monkeypatch.setattr(C, "RAW", tmp_path / "raw")
    monkeypatch.setattr(C, "MANIFEST", tmp_path / "manifest.csv")
    assert len(acquire.ingest(tmp_path / "in")) == 1
    (tmp_path / "in" / "Data/Raw/Lahore" / "lahore_Gulberg.csv").write_bytes(content + b"x")
    monkeypatch.setattr(C, "RAW", tmp_path / "raw2")
    with pytest.raises(acquire.RawIntegrityError):
        acquire.ingest(tmp_path / "in")


def test_load_manifest_collapses_duplicate_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MANIFEST", tmp_path / "manifest.csv")
    row_ = {"file": "L/a.csv", "station_id": "l__a", "sha256": "h"}
    pd.DataFrame([row_, row_]).to_csv(C.MANIFEST, index=False)
    assert len(acquire.load_manifest()) == 1 and len(pd.read_csv(C.MANIFEST)) == 1


def test_annotate_updates_manifest_without_touching_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MANIFEST", tmp_path / "manifest.csv")
    pd.DataFrame([{"file": "L/a.csv", "station_id": "l__a", "sha256": "h", "fetch_time_utc": "x", "fetch_time_basis": "y"}]).to_csv(C.MANIFEST, index=False)
    out = acquire.annotate("2026-10-20T18:30+05:00", "recorded by team member", "received from a team member")
    assert out.loc[0, "fetch_time_utc"] == "2026-10-20T13:30:00+00:00" and out.loc[0, "sha256"] == "h"
    assert out.loc[0, "provenance"] == "received from a team member"
    with pytest.raises(ValueError):
        acquire.annotate("2026-10-20T18:30")
