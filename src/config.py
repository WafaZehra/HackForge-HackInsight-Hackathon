from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INCOMING = DATA / "incoming"
RAW = DATA / "raw"
CLEANED = DATA / "cleaned"
PROCESSED = DATA / "processed"
STAGING = DATA / "_staging"
MANIFEST = ROOT / "manifest.csv"
LOGS = ROOT / "logs"
RUN_LOG = LOGS / "pipeline_runs.jsonl"
WAREHOUSE_NAME = "warehouse.duckdb"

SOURCE_NAME = "Open-Meteo Air Quality API"
PUBLISHER = "Open-Meteo (model data: Copernicus CAMS global atmospheric composition)"
SOURCE_URL = "https://open-meteo.com/en/docs/air-quality-api"
LICENCE = "CC BY 4.0 (Open-Meteo; attribution required)"
API_ENDPOINT = "https://air-quality-api.open-meteo.com/v1/air-quality"
API_HOURLY_VARS = ("pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone", "sulphur_dioxide")

VAR_MAP = {
    "pm10": "pm10",
    "pm2_5": "pm25",
    "carbon_monoxide": "co",
    "nitrogen_dioxide": "no2",
    "ozone": "o3",
    "sulphur_dioxide": "so2",
}
POLLUTANTS = ("pm10", "pm25", "co", "no2", "o3", "so2")
CANONICAL_UNIT = "μg/m³"
UNIT_TO_UG_M3 = {"μg/m³": 1.0, "mg/m³": 1000.0, "ng/m³": 0.001}

PLAUSIBLE_MAX = {"pm10": 2000.0, "pm25": 1000.0, "co": 50000.0, "no2": 1000.0, "o3": 1000.0, "so2": 1000.0}
MAX_FILL_HOURS = 3
MIN_HOURS_PER_DAY = 18
PKT_OFFSET_HOURS = 5
CITY_FIXES = {"Peshawer": "Peshawar"}

WHO_2021_24H = {"pm25": 15.0, "pm10": 45.0}

MIN_TOTAL_ROWS = 50_000
MIN_HOURS_PER_STATION = 2 * 365 * 24

SMOOTH_QUANTILE = 0.95
SPIKE_WINDOW = 25
SPIKE_K = 12.0
STALE_GRACE_HOURS = 6
PSI_THRESHOLD = 0.25
PSI_MIN_OBS = 500

MODEL_DIR = DATA / "model"
SEED = 42
HORIZON_DAYS = 1
BASELINE = "persistence_1d"
TEST_DAYS = 90
CV_FOLDS = 4
CV_BLOCK_DAYS = 60
INTERVAL_ALPHA = 0.2
ALERT_THRESHOLD = 37.5
BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK_DAYS = 7
HGB_GRID = [
    {"learning_rate": lr, "max_depth": d, "max_iter": it, "min_samples_leaf": leaf, "l2_regularization": 1.0}
    for lr in (0.03, 0.06) for d in (3, 5) for it in (150, 300) for leaf in (20,)
]
