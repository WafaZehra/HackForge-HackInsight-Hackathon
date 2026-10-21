from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from . import config as C
from .features import FEATURES, META

DROP_FEATURES = [f for f in FEATURES if f not in META]

SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


@dataclass(frozen=True)
class Split:
    last_target: pd.Timestamp
    test_start: pd.Timestamp
    blocks: list[tuple[pd.Timestamp, pd.Timestamp]]


def make_split(df: pd.DataFrame) -> Split:
    last = df.loc[df["y"].notna(), "target_date"].max()
    test_start = last - pd.Timedelta(days=C.TEST_DAYS - 1)
    blocks, end = [], test_start
    for _ in range(C.CV_FOLDS):
        start = end - pd.Timedelta(days=C.CV_BLOCK_DAYS)
        blocks.append((start, end))
        end = start
    return Split(last, test_start, blocks[::-1])


def train_rows(df: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    return df[(df["target_date"] < before) & df["y"].notna()]


def window_rows(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None = None) -> pd.DataFrame:
    mask = (df["target_date"] >= start) & df["y"].notna()
    if end is not None:
        mask &= df["target_date"] < end
    return df[mask]


def make_model(params: dict, loss: str = "absolute_error", quantile: float | None = None) -> HistGradientBoostingRegressor:
    kw = dict(loss=loss, random_state=C.SEED, categorical_features="from_dtype", early_stopping=False, **params)
    if quantile is not None:
        kw["quantile"] = quantile
    return HistGradientBoostingRegressor(**kw)


def predict_clipped(model, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(frame[FEATURES]), 0.0, None)


def tune(df: pd.DataFrame, split: Split) -> tuple[dict, pd.DataFrame]:
    rows = []
    for i, params in enumerate(C.HGB_GRID):
        maes = []
        for start, end in split.blocks:
            tr, te = train_rows(df, start), window_rows(df, start, end)
            model = make_model(params).fit(tr[FEATURES], tr["y"])
            maes.append(float(np.mean(np.abs(te["y"].to_numpy() - predict_clipped(model, te)))))
        rows.append({"grid_id": i, **params, "cv_mae": float(np.mean(maes)), **{f"fold{k}_mae": m for k, m in enumerate(maes)}})
    table = pd.DataFrame(rows)
    best = table.sort_values(["cv_mae", "grid_id"]).iloc[0]
    return C.HGB_GRID[int(best["grid_id"])], table


def out_of_fold(df: pd.DataFrame, split: Split, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts, importance = [], pd.DataFrame()
    lo_q, hi_q = C.INTERVAL_ALPHA / 2, 1 - C.INTERVAL_ALPHA / 2
    for k, (start, end) in enumerate(split.blocks):
        tr, te = train_rows(df, start), window_rows(df, start, end)
        point = make_model(params).fit(tr[FEATURES], tr["y"])
        lo = make_model(params, "quantile", lo_q).fit(tr[FEATURES], tr["y"])
        hi = make_model(params, "quantile", hi_q).fit(tr[FEATURES], tr["y"])
        part = te.drop(columns=DROP_FEATURES, errors="ignore").copy()
        part["pred"] = predict_clipped(point, te)
        part["lo_raw"] = predict_clipped(lo, te)
        part["hi_raw"] = predict_clipped(hi, te)
        part["fold"] = k
        parts.append(part)
        if k == len(split.blocks) - 1:
            r = permutation_importance(point, te[FEATURES], te["y"], scoring="neg_mean_absolute_error",
                                       n_repeats=5, random_state=C.SEED)
            importance = pd.DataFrame({"feature": FEATURES, "mae_increase": r.importances_mean,
                                       "std": r.importances_std}).sort_values("mae_increase", ascending=False)
    return pd.concat(parts, ignore_index=True), importance.reset_index(drop=True)


def cqr_offset(oof: pd.DataFrame) -> float:
    lo, hi = np.minimum(oof["lo_raw"], oof["hi_raw"]), np.maximum(oof["lo_raw"], oof["hi_raw"])
    scores = np.maximum(lo - oof["y"], oof["y"] - hi).to_numpy()
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1 - C.INTERVAL_ALPHA)) / n)
    return float(np.quantile(scores, level, method="higher"))


def apply_interval(pred: np.ndarray, lo_raw: np.ndarray, hi_raw: np.ndarray, qhat: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.minimum(lo_raw, hi_raw) - qhat
    hi = np.maximum(lo_raw, hi_raw) + qhat
    return np.clip(np.minimum(lo, pred), 0.0, None), np.maximum(hi, pred)


def fit_final(train: pd.DataFrame, params: dict) -> dict:
    lo_q, hi_q = C.INTERVAL_ALPHA / 2, 1 - C.INTERVAL_ALPHA / 2
    return {
        "point": make_model(params).fit(train[FEATURES], train["y"]),
        "lo": make_model(params, "quantile", lo_q).fit(train[FEATURES], train["y"]),
        "hi": make_model(params, "quantile", hi_q).fit(train[FEATURES], train["y"]),
    }


def predict_with_interval(models: dict, frame: pd.DataFrame, qhat: float) -> pd.DataFrame:
    pred = predict_clipped(models["point"], frame)
    lo_raw, hi_raw = predict_clipped(models["lo"], frame), predict_clipped(models["hi"], frame)
    lo, hi = apply_interval(pred, lo_raw, hi_raw, qhat)
    out = frame.drop(columns=DROP_FEATURES, errors="ignore").copy()
    out["pred"], out["lo_raw"], out["hi_raw"], out["lo"], out["hi"] = pred, lo_raw, hi_raw, lo, hi
    return out


def evaluation_set(frame: pd.DataFrame) -> pd.DataFrame:
    keep = frame["y"].notna() & frame["base_persistence_1d"].notna() & frame["base_seasonal_7d"].notna() & frame["pred"].notna()
    return frame[keep].copy()


def alert_scores(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[float, float, float]:
    actual, flagged = y >= thr, p >= thr
    tp, fp, fn = int((actual & flagged).sum()), int((~actual & flagged).sum()), int((actual & ~flagged).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp and precision == precision and recall == recall else float("nan")
    return precision, recall, f1


def summarize(frame: pd.DataFrame) -> dict:
    y = frame["y"].to_numpy(dtype=float)
    out = {"n": len(frame), "mean_actual": float(y.mean())}
    for name, col in (("baseline", "base_persistence_1d"), ("seasonal7", "base_seasonal_7d"), ("model", "pred")):
        p = frame[col].to_numpy(dtype=float)
        out[f"mae_{name}"] = float(np.abs(y - p).mean())
        out[f"rmse_{name}"] = float(np.sqrt(((y - p) ** 2).mean()))
        out[f"bias_{name}"] = float((p - y).mean())
    out["skill_vs_baseline"] = 1 - out["mae_model"] / out["mae_baseline"] if out["mae_baseline"] else float("nan")
    out["rel_mae_model"] = out["mae_model"] / out["mean_actual"] if out["mean_actual"] else float("nan")
    out["coverage"] = float(((y >= frame["lo"]) & (y <= frame["hi"])).mean())
    out["coverage_raw"] = float(((y >= frame["lo_raw"]) & (y <= frame["hi_raw"])).mean())
    out["mean_width"] = float((frame["hi"] - frame["lo"]).mean())
    for name, col in (("baseline", "base_persistence_1d"), ("model", "pred")):
        pr, rc, f1 = alert_scores(y, frame[col].to_numpy(dtype=float), C.ALERT_THRESHOLD)
        out[f"alert_precision_{name}"], out[f"alert_recall_{name}"], out[f"alert_f1_{name}"] = pr, rc, f1
    out["alert_days"] = int((y >= C.ALERT_THRESHOLD).sum())
    return out


def group_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = [{by: key, **summarize(g)} for key, g in frame.groupby(by, observed=True, sort=True)]
    return pd.DataFrame(rows)


def level_label(y: pd.Series) -> pd.Series:
    return pd.cut(y, [-np.inf, C.WHO_2021_24H["pm25"], C.ALERT_THRESHOLD, np.inf], right=False,
                  labels=["below WHO guideline", "above guideline, below alert", "alert level"]).astype(str)


def block_bootstrap(frame: pd.DataFrame) -> dict:
    daily = frame.assign(eb=(frame["y"] - frame["base_persistence_1d"]).abs(), em=(frame["y"] - frame["pred"]).abs())
    daily = daily.groupby("target_date")[["eb", "em"]].mean().sort_index()
    eb, em = daily["eb"].to_numpy(), daily["em"].to_numpy()
    n, block = len(eb), min(C.BOOTSTRAP_BLOCK_DAYS, len(eb))
    rng = np.random.default_rng(C.SEED)
    n_blocks = int(math.ceil(n / block))
    diffs, skills = [], []
    for _ in range(C.BOOTSTRAP_N):
        idx = (rng.integers(0, n - block + 1, n_blocks)[:, None] + np.arange(block)).ravel()[:n]
        diffs.append(eb[idx].mean() - em[idx].mean())
        skills.append(1 - em[idx].mean() / eb[idx].mean())
    d, s = np.array(diffs), np.array(skills)
    return {"days": n, "mae_gain_mean": float(eb.mean() - em.mean()), "mae_gain_ci_low": float(np.percentile(d, 2.5)),
            "mae_gain_ci_high": float(np.percentile(d, 97.5)), "skill_ci_low": float(np.percentile(s, 2.5)),
            "skill_ci_high": float(np.percentile(s, 97.5)), "share_resamples_model_better": float((d > 0).mean())}
