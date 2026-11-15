"""Settings for the Taiwan Grass-Owl analysis. Every judgement call lives here.

Change a value here, rerun the scripts, and every table and plot follows.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOURLY_CSV = ROOT / "data/processed/grassowl_hourly.csv"   # built by scripts/clean_grassowl.py
OUT = ROOT / "outputs/taiwan"

# --- What counts as an owl detection (flaw 5) ---------------------------------
# "recorder": hours the recorder detected an owl. The recorder can reach 100% of these.
# "any":      hours either sensor detected an owl. The recorder alone can reach at most
#             recorder-hours / any-hours (about 95%), because camera-only hours are invisible to it.
TARGET = "recorder"

# --- Simple rule baseline (flaw 1) --------------------------------------------
# The data only covers 17:00-06:00, so "record at night only" changes nothing.
# Instead the rule skips the dusk and dawn edge hours, where the recorder rarely detects owls.
EDGE_HOURS = [17, 6]

# --- Season (used by the context selector) ------------------------------------
# Southwestern Taiwan: wet season May-Oct, dry season Nov-Apr (assumption; check a local source).
WET_MONTHS = [5, 6, 7, 8, 9, 10]

# --- Recording schedule in the data (flaw 3) ----------------------------------
# Effort-standardised files: recorder runs 10 min per sampled hour, 14 sampled hours per night
# (17:00-06:00). Savings are measured against this schedule, not continuous recording.
MINUTES_PER_SAMPLED_HOUR = 10
SAMPLED_HOURS_PER_NIGHT = 14

# --- AudioMoth power (flaw 4) -------------------------------------------------
# Hill et al. 2018, Methods Ecol Evol 9:1199-1211, doi:10.1111/2041-210X.12955:
#   ~23 mW average while recording at 48 kHz (case study 1, 16 GB card); 80 uW asleep.
# The paper gives NO battery-life days, so days below depend on BATTERY_WH (an assumption).
RECORD_MW = 23.0
SLEEP_MW = 0.08
# Assumption: 3 x lithium AA (e.g. Energizer L91, ~3.0 Ah at 1.5 V nominal each) = 13.5 Wh.
# Real usable energy is lower (cold, cut-off voltage); treat days as optimistic.
BATTERY_WH = 13.5

# --- Context selector ----------------------------------------------------------
# The camera is always on (motion-triggered, cheap when idle, per plan v2), so the selector
# only schedules the recorder. The headline measure is SYSTEM-LEVEL detection: of all owl-hours
# (either sensor), the share caught by the camera plus the recorder in the hours it is switched on.
# The selector is trained to predict hours where the recorder hears an owl the camera missed,
# because those are the only hours the recorder adds anything.
PROMISED_DETECTION = 0.80      # cutoff is set so this share of owl-hours is caught on known data
LEARNING_MONTHS = 3            # months of labelled data at the new place before the schedule is frozen
MIN_LEARNING_POSITIVES = 5     # fewer recorder-adds hours than this: set the cutoff on the home region
LOCAL_WEIGHT = "balance"       # learning months weighted to count as much as the whole home region

# --- Plot colours (validated categorical palette, light mode) -----------------
COLOURS = {
    "both": "#2a78d6",           # blue
    "recorder_only": "#1baf7a",  # aqua
    "camera_only": "#eb6834",    # orange
    "model": "#2a78d6",
    "rule": "#eb6834",
    "oracle": "#52514e",         # reference lines in neutral ink
    "random": "#8a8984",
    "home_only": "#2a78d6",      # blue
    "home_local": "#eb6834",     # orange
    "camera": "#8a8984",
}
