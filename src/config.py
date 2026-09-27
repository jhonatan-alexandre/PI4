"""Central configuration for the LSTM energy-consumption forecasting pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Dataset ---------------------------------------------------------------
UCI_DATASET_ID = 235  # Individual Household Electric Power Consumption
UCI_TOTAL_ROWS = 2_075_259
RAW_CSV_PATH = RAW_DATA_DIR / "household_power_consumption.csv"
CLEAN_PARQUET_PATH = PROCESSED_DATA_DIR / "household_power_consumption_clean.parquet"

FEATURE_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
TARGET_COLUMN = "Global_active_power"

# --- Windowing ---------------------------------------------------------------
# The raw data is sampled every minute; each horizon is an interval for energy accumulation.
HORIZONS = {"15min": 15, "30min": 30, "1h": 60}
LOOKBACK_MINUTES = 120  # length of the input window (past observations) fed to the LSTM
WINDOW_STRIDE_MINUTES = 5

# --- Splits ---------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # remaining (0.15) goes to test, splits are chronological (no shuffling)

# Set to an int to only use the most recent N rows (useful for fast experimentation).
# Use None to train on the full ~4 years / 2M rows series.
MAX_ROWS = 200_000

# --- Model / training ---------------------------------------------------------------
BATCH_SIZE = 256
EPOCHS = 30
LSTM_UNITS = (64, 32)
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
