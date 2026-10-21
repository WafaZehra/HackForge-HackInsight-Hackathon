from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import modeling as M
from src.features import FEATURES, build_supervised


def synthetic_daily(n_days=400, stations=("a__s1", "b__s2"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    frames = []
    for i, sid in enumerate(stations):
        base = 30 + 20 * np.sin(np.arange(n_days) / 58.0) + rng.normal(0, 4, n_days)
        frames.append(pd.DataFrame({
            "station_id": sid, "city": sid.split("__")[0].upper(), "date": dates, "pm25_mean_24h": np.clip(base, 1, None),
            "pm10_mean_24h": np.clip(base * 1.8, 1, None), "no2_mean_24h": 20.0, "so2_mean_24h": 5.0, "o3_mean_24h": 40.0,
            "co_mean_24h": 300.0, "pm25_max_1h": np.clip(base * 1.4, 1, None), "latitude": 30.0 + i, "longitude": 70.0 + i,
            "elevation_m": 100.0 * (i + 1)}))
    return pd.concat(frames, ignore_index=True)


def test_features_never_use_future_values():
    daily = synthetic_daily()
    cutoff = pd.Timestamp("2025-09-01")
    altered = daily.copy()
    mask = altered["date"] > cutoff
    for col in ("pm25_mean_24h", "pm10_mean_24h", "no2_mean_24h", "so2_mean_24h", "o3_mean_24h", "co_mean_24h", "pm25_max_1h"):
        altered.loc[mask, col] = altered.loc[mask, col] * 7 + 100
    a, b = build_supervised(daily), build_supervised(altered)
    past = a["date"] <= cutoff
    pd.testing.assert_frame_equal(a.loc[past, FEATURES].reset_index(drop=True), b.loc[past, FEATURES].reset_index(drop=True))


def test_target_is_next_day_and_baseline_is_today():
    sup = build_supervised(synthetic_daily())
    one = sup[sup["station_id"] == "a__s1"].reset_index(drop=True)
    assert (one["target_date"] == one["date"] + pd.Timedelta(days=1)).all()
    assert one.loc[10, "y"] == one.loc[11, "pm25_d0"]
    assert (one["base_persistence_1d"].dropna() == one["pm25_d0"].dropna()).all()
    assert one.loc[20, "base_seasonal_7d"] == one.loc[14, "pm25_d0"]
    assert one.loc[10, "target_row_id"] == one.loc[11, "feature_row_id"]


def test_split_is_chronological_and_blocks_precede_test():
    sup = build_supervised(synthetic_daily())
    split = M.make_split(sup)
    assert split.last_target - split.test_start == pd.Timedelta(days=C.TEST_DAYS - 1)
    assert split.blocks[-1][1] == split.test_start
    for (s1, e1), (s2, e2) in zip(split.blocks[:-1], split.blocks[1:]):
        assert e1 == s2 and s1 < e1
    train, test = M.train_rows(sup, split.test_start), M.window_rows(sup, split.test_start)
    assert train["target_date"].max() < test["target_date"].min()
    assert train["date"].max() < split.test_start


def test_conformal_offset_restores_coverage():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 2000)
    oof = pd.DataFrame({"y": y, "lo_raw": np.full(2000, -0.3), "hi_raw": np.full(2000, 0.3)})
    qhat = M.cqr_offset(oof)
    lo, hi = M.apply_interval(np.zeros(2000), oof["lo_raw"].to_numpy(), oof["hi_raw"].to_numpy(), qhat)
    fresh = rng.normal(0, 1, 5000)
    inside = ((fresh >= -0.3 - qhat) & (fresh <= 0.3 + qhat)).mean()
    assert qhat > 0 and abs(inside - (1 - C.INTERVAL_ALPHA)) < 0.03


def test_interval_always_contains_point_and_is_nonnegative():
    pred = np.array([5.0, 0.2, 50.0])
    lo, hi = M.apply_interval(pred, np.array([6.0, 0.0, 40.0]), np.array([7.0, 0.1, 45.0]), 0.0)
    assert (lo <= pred).all() and (hi >= pred).all() and (lo >= 0).all()


def test_alert_scores_hand_computed():
    y = np.array([50, 50, 10, 10, 40.0])
    p = np.array([60, 10, 60, 10, 38.0])
    precision, recall, f1 = M.alert_scores(y, p, 37.5)
    assert precision == pytest.approx(2 / 3) and recall == pytest.approx(2 / 3) and f1 == pytest.approx(2 / 3)


def test_alert_scores_handle_no_events():
    precision, recall, f1 = M.alert_scores(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 37.5)
    assert np.isnan(precision) and np.isnan(recall) and np.isnan(f1)


def test_summarize_skill_and_bootstrap_on_known_errors():
    n = 120
    dates = pd.date_range("2026-01-01", periods=n)
    frame = pd.DataFrame({"target_date": dates, "y": 50.0, "base_persistence_1d": 60.0, "base_seasonal_7d": 70.0,
                          "pred": 55.0, "lo": 45.0, "hi": 65.0, "lo_raw": 45.0, "hi_raw": 65.0})
    s = M.summarize(frame)
    assert s["mae_baseline"] == 10 and s["mae_model"] == 5 and s["skill_vs_baseline"] == pytest.approx(0.5)
    assert s["coverage"] == 1.0
    boot = M.block_bootstrap(frame)
    assert boot["mae_gain_ci_low"] == pytest.approx(5.0) and boot["share_resamples_model_better"] == 1.0


def test_small_end_to_end_run_produces_valid_intervals():
    sup = build_supervised(synthetic_daily(n_days=500))
    split = M.make_split(sup)
    params = C.HGB_GRID[0]
    models = M.fit_final(M.train_rows(sup, split.test_start), params)
    test = M.window_rows(sup, split.test_start)
    out = M.evaluation_set(M.predict_with_interval(models, test, 0.0))
    assert len(out) > 0 and out["pred"].notna().all() and (out["lo"] <= out["pred"]).all()
