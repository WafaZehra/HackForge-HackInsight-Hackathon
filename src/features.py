from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import config as C

NUMERIC_FEATURES = [
    "pm25_d0", "pm25_d1", "pm25_d2", "pm25_d3", "pm25_d6",
    "pm25_mean3", "pm25_mean7", "pm25_mean14", "pm25_mean30", "pm25_std7", "pm25_trend",
    "pm25_max1h_d0", "pm10_d0", "no2_d0", "so2_d0", "o3_d0", "co_d0", "pm10_pm25_ratio",
    "dow", "doy_sin", "doy_cos", "elevation_m", "latitude", "longitude",
]
CATEGORICAL_FEATURES = ["city"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
META = ["station_id", "city", "date", "target_date", "feature_row_id", "target_row_id", "y", "base_persistence_1d", "base_seasonal_7d"]


def load_daily(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT g.station_id, g.city, g.date_pkt AS date, g.pm25_mean_24h, g.pm10_mean_24h, g.no2_mean_24h,
                   g.so2_mean_24h, g.o3_mean_24h, g.co_mean_24h, g.pm25_max_1h,
                   d.latitude, d.longitude, d.elevation_m
            FROM gold_station_daily g JOIN dim_station d ON g.station_id = d.station_id
            ORDER BY g.station_id, g.date_pkt
        """).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _station_features(g: pd.DataFrame) -> pd.DataFrame:
    sid, city = g["station_id"].iloc[0], g["city"].iloc[0]
    full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
    g = g.set_index("date").reindex(full)
    p = g["pm25_mean_24h"]
    out = pd.DataFrame(index=full)
    out["pm25_d0"] = p
    for k in (1, 2, 3, 6):
        out[f"pm25_d{k}"] = p.shift(k)
    for w in (3, 7, 14, 30):
        out[f"pm25_mean{w}"] = p.rolling(w, min_periods=max(2, int(0.6 * w))).mean()
    out["pm25_std7"] = p.rolling(7, min_periods=4).std()
    out["pm25_trend"] = p - p.shift(1)
    out["pm25_max1h_d0"] = g["pm25_max_1h"]
    for v in ("pm10", "no2", "so2", "o3", "co"):
        out[f"{v}_d0"] = g[f"{v}_mean_24h"]
    out["pm10_pm25_ratio"] = g["pm10_mean_24h"] / p.where(p > 0)
    target_date = full + pd.Timedelta(days=C.HORIZON_DAYS)
    out["target_date"] = target_date
    out["dow"] = target_date.dayofweek
    doy = target_date.dayofyear.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["elevation_m"] = float(g["elevation_m"].dropna().iloc[0])
    out["latitude"] = float(g["latitude"].dropna().iloc[0])
    out["longitude"] = float(g["longitude"].dropna().iloc[0])
    out["y"] = p.shift(-C.HORIZON_DAYS)
    out["base_persistence_1d"] = p
    out["base_seasonal_7d"] = p.shift(6)
    out["station_id"] = sid
    out["city"] = city
    out["feature_row_id"] = [f"{sid}:{d:%Y%m%d}" for d in full]
    out["target_row_id"] = [f"{sid}:{d:%Y%m%d}" for d in target_date]
    out.index.name = "date"
    return out.reset_index()


def build_supervised(daily: pd.DataFrame) -> pd.DataFrame:
    parts = [_station_features(g) for _, g in daily.groupby("station_id", sort=True)]
    df = pd.concat(parts, ignore_index=True)
    df["city"] = pd.Categorical(df["city"], categories=sorted(daily["city"].unique()))
    return df.sort_values(["station_id", "date"]).reset_index(drop=True)
