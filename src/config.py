"""Central configuration: paths, source metadata, units, thresholds.

Everything tunable lives here so the pipeline code has no magic numbers.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INCOMING = DATA / "incoming"      # drop Data.zip (or the extracted Raw/ folder) here once
RAW = DATA / "raw"                # layer 1: byte-identical copies, never edited
CLEANED = DATA / "cleaned"        # layer 2: typed, UTC, regularised, flagged
PROCESSED = DATA / "processed"    # layer 3: DuckDB warehouse (gold tables)
STAGING = DATA / "_staging"       # layers are built here, promoted only if validation passes
MANIFEST = ROOT / "manifest.csv"
LOGS = ROOT / "logs"
RUN_LOG = LOGS / "pipeline_runs.jsonl"
WAREHOUSE_NAME = "warehouse.duckdb"

# ---- Source metadata (written into manifest.csv) -------------------------------------------
SOURCE_NAME = "Open-Meteo Air Quality API"
PUBLISHER = "Open-Meteo (model data: Copernicus CAMS global atmospheric composition)"
SOURCE_URL = "https://open-meteo.com/en/docs/air-quality-api"
# Verify at https://open-meteo.com/en/license before submitting; attribution is required.
LICENCE = "CC BY 4.0 (Open-Meteo; attribution required)"
API_ENDPOINT = "https://air-quality-api.open-meteo.com/v1/air-quality"
API_HOURLY_VARS = ("pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone", "sulphur_dioxide")

# ---- Variables and units -----------------------------------------------------------------
# Open-Meteo column name -> canonical short name used everywhere downstream.
VAR_MAP = {
    "pm10": "pm10",
    "pm2_5": "pm25",
    "carbon_monoxide": "co",
    "nitrogen_dioxide": "no2",
    "ozone": "o3",
    "sulphur_dioxide": "so2",
}
POLLUTANTS = ("pm10", "pm25", "co", "no2", "o3", "so2")
CANONICAL_UNIT = "μg/m³"  # U+03BC; the micro sign U+00B5 is normalised to this before comparing
UNIT_TO_UG_M3 = {"μg/m³": 1.0, "mg/m³": 1000.0, "ng/m³": 0.001}

# ---- Cleaning rules -------------------------------------------------------------------------
# Above these values a reading is flagged (kept, never guessed at). Negatives are set to NaN.
PLAUSIBLE_MAX = {"pm10": 2000.0, "pm25": 1000.0, "co": 50000.0, "no2": 1000.0, "o3": 1000.0, "so2": 1000.0}
MAX_FILL_HOURS = 3          # interior gaps up to this long are linearly interpolated and marked
MIN_HOURS_PER_DAY = 18      # a PKT day needs >= 18 valid hours (75%) to get a daily mean
PKT_OFFSET_HOURS = 5        # Pakistan Standard Time = UTC+5, no DST. Used for display/daily bins only.
CITY_FIXES = {"Peshawer": "Peshawar"}  # folder-name typos in the supplied files

# ---- Reference thresholds (WHO Global Air Quality Guidelines 2021, 24-hour, ug/m3) ---------
WHO_2021_24H = {"pm25": 15.0, "pm10": 45.0}

# ---- Volume rule from the event guide ---------------------------------------------------------
MIN_TOTAL_ROWS = 50_000
MIN_HOURS_PER_STATION = 2 * 365 * 24
