# Shared paths and settings for all scripts
import os
from pathlib import Path

# Data folder; set SAFARI_ROOT to run from another location
ROOT = Path(os.environ.get("SAFARI_ROOT", r"C:\College\AFML\safari"))

ANN_TEST = ROOT / "sa_fari_test_ext.json"
ANN_TRAIN = ROOT / "sa_fari_train_ext.json"
VIDEO_DIR = ROOT / "test_videos"
OUT = ROOT / "outputs"
RESULTS = OUT / "results"

SEED = 42

RESULTS.mkdir(parents=True, exist_ok=True)
