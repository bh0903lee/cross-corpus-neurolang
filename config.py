"""
config.py — Paths and hyperparameters.
BASE_DIR resolves from the DATA_DIR environment variable if set
(e.g. export DATA_DIR=/path/to/data_root), else from this file's location.
"""
import os
from pathlib import Path

# ─── BASE_DIR auto-detection ─────────────────────────────────────────────────
_THIS_POC_DIR = Path(__file__).resolve().parent   # poc/
_AUTO_BASE    = _THIS_POC_DIR.parent              # TalkBank_Research/

_env_data = os.environ.get("DATA_DIR")
if _env_data:
    _env_path = Path(_env_data)
    if _env_path.exists():
        BASE_DIR = _env_path
    else:
        import warnings
        warnings.warn(
            f"DATA_DIR='{_env_data}' does not exist. "
            f"Falling back to auto-detected path: {_AUTO_BASE}",
            UserWarning,
            stacklevel=2,
        )
        BASE_DIR = _AUTO_BASE
else:
    BASE_DIR = _AUTO_BASE

POC_DIR = BASE_DIR / "poc"

# OUTPUT_DIR env var overrides the default poc/results/
RESULTS_DIR = Path(os.environ.get("OUTPUT_DIR", str(POC_DIR / "results")))

# RESULTS_DIR is not created at import time; entry scripts create it when needed.

APHASIA_TEXT_DIR = BASE_DIR / "Aphasia_2TrainTest" / "aphasiabank_text_P"
AD_TEST_DIR      = BASE_DIR / "AD_2TrainTest" / "ADReSSo" / "test"
AD_TRAIN_DIR     = BASE_DIR / "AD_2TrainTest" / "ADReSSo" / "train"
METADATA_CSV     = BASE_DIR / "participant_task_coverage.csv"
AD_LABELS_CSV    = BASE_DIR / "example_metadata.csv"
AD_TRAIN_META_CSV = BASE_DIR / "AD_2TrainTest" / "ADReSSo" / "metadata-train.csv"
AD_TEST_META_CSV  = BASE_DIR / "AD_2TrainTest" / "ADReSSo" / "metadata-test.csv"

# ─── Additional dataset paths ────────────────────────────────────────────────
TBI_CTRL_DIR = BASE_DIR / "TBI_1BeforeSplit" / "Coelho" / "Control"
TBI_TBI_DIR  = BASE_DIR / "TBI_1BeforeSplit" / "Coelho" / "TBI"
RHD_CTRL_DIR = BASE_DIR / "RHD_1BeforeSplit" / "English_txt" / "Control"
RHD_RHD_DIR  = BASE_DIR / "RHD_1BeforeSplit" / "English_txt" / "RHD"
MCI_CTRL_DIR = BASE_DIR / "MCI_1BeforeSplit" / "Delaware_txt" / "Control"
MCI_MCI_DIR  = BASE_DIR / "MCI_1BeforeSplit" / "Delaware_txt" / "MCI"

# ─── Demographics paths (TBI / RHD / MCI) ────────────────────────────────────
TBI_TBI_DEMO_XLSX  = BASE_DIR / "TBI_1BeforeSplit"  / "Coelho"   / "demo_TBI.xlsx"
TBI_CTRL_DEMO_XLSX = BASE_DIR / "TBI_1BeforeSplit"  / "Coelho"   / "demo_Control.xlsx"
RHD_CTRL_DEMO_XLSX = BASE_DIR / "RHD_1BeforeSplit"  / "English"  / "Control" / "0demo.xlsx"
RHD_RHD_DEMO_XLSX  = BASE_DIR / "RHD_1BeforeSplit"  / "English"  / "RHD"     / "0demo.xlsx"
MCI_DEMO_XLSX      = BASE_DIR / "MCI_1BeforeSplit"  / "Delaware" / "demo-test.xlsx"

# ─── Audio data paths ────────────────────────────────────────────────────────
AUDIO_BASE_DIR = BASE_DIR / "Dataset_Audio"

# ─── Labels ──────────────────────────────────────────────────────────────────
LABEL_MAP    = {"Control": 0, "Aphasia": 1, "AD": 2, "TBI": 3, "RHD": 4, "MCI": 5}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
N_CLASSES    = 6

CONTROL_WAB_TYPES = {"control", "notaphasicbywab", "Control", "NotAphasicByWAB"}

# ─── Feature vector dimensions ───────────────────────────────────────────────
N_LING_FEATURES = 12
N_TAB_FEATURES  = 4

# ─── BERT settings ───────────────────────────────────────────────────────────
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_DIM        = 768

# ─── wav2vec2 settings ───────────────────────────────────────────────────────
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-large"
AUDIO_DIM          = 1024
# Segmentation: see SEGMENT_SECONDS=30 in audio_extractor.py
# (audio split into 30-s segments → segment-level mean pooling)

# ─── Model architecture ──────────────────────────────────────────────────────
TEXT_ENC_DIM     = 64
TAB_ENC_DIM      = 32
HIDDEN_DIM       = 128
N_QUERY_TOKENS   = 16
N_ATTN_HEADS     = 4
N_QFORMER_LAYERS = 2

MEMORY_SIZE = 128
MEMORY_DIM  = HIDDEN_DIM

# ─── Training settings ───────────────────────────────────────────────────────
BATCH_SIZE = 32
LR         = 2e-4
N_EPOCHS   = 150
SEED       = 42
VAL_RATIO  = 0.15

LAMBDA_CLS      = 1.0
LAMBDA_REG      = 0.05
LAMBDA_SIM      = 0.1
LABEL_SMOOTHING = 0.05
FOCAL_GAMMA     = 2.0
WARMUP_EPOCHS   = 15

# Class-imbalance correction strengths
FOCAL_ALPHA_POWER = 1.0
SAMPLER_POWER     = 0.7
