# Simultaneous Multi-Task Assessment of Five Neurological Language Disorders via Cross-Corpus Multimodal Learning

Code accompanying the manuscript. A unified multimodal Q-Former model classifies
five neurological language disorders (Aphasia, AD, MCI, TBI, RHD) plus a healthy
Control class across five TalkBank corpora, with auxiliary heads for WAB-AQ
severity regression, similarity retrieval, and longitudinal progression.

## Data access

The corpora are **not** redistributed with this code. They must be obtained
directly from TalkBank (https://talkbank.org) after registration and agreement
to the corpus-specific usage terms:

- AphasiaBank (Aphasia / Control)
- ADReSSo 2021 challenge data (AD / Control; audio only — transcripts are
  generated with OpenAI Whisper)
- TBIBank Coelho (TBI / Control)
- RHDBank English (RHD / Control)
- DementiaBank Delaware (MCI / Control)

`config.py` expects the corpora under a common data root; set the `DATA_DIR`
environment variable or place this directory next to the data root (see
`config.py` for the expected sub-directory layout).

In addition to the transcripts and audio, the loader reads (i) the demographic
spreadsheets distributed with the Coelho, RHDBank, and Delaware corpora, and
(ii) an AphasiaBank participant index CSV (`participant_task_coverage.csv`,
one row per participant with columns `participant_id, corpus, age, sex,
wab_type, wab_aq, ...`) compiled from the AphasiaBank demographic
spreadsheets available to registered TalkBank members. These files contain
participant-level information and are therefore not redistributed here.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11; a single GPU (>=16 GB) is sufficient — only a ~1.1M-parameter head
is trained on top of frozen encoders.
Decoding mp3 corpora additionally requires `ffmpeg` on the system path
(used by `pydub`).

## Pipeline

1. **Feature pre-computation** (one-time; cached under `results/`):
   - `python extract_text_features.py` — frozen RoBERTa-large [CLS] features.
   - wav2vec2-large segment features and paralinguistic features are computed
     and cached automatically on first data load (`audio_extractor.py`,
     `paralinguistic_extractor.py`).
2. **Main experiments** (proposed model, ablations, LOCO; five seeds each):
   ```
   python run_rl_all_experiments.py --worker --exp proposed --device cuda:0
   python run_rl_all_experiments.py --worker --exp LOCO_ADReSSo --device cuda:0
   python run_rl_all_experiments.py --worker --exp A_notab --device cuda:0
   python run_rl_all_experiments.py --worker --exp E5_Richardson --device cuda:0
   ```
   See the `EXPERIMENTS` dict in `run_rl_all_experiments.py` for all
   configurations (audio variants, LOCO, tabular ablation, sub-corpus
   hold-outs). Registry keys map to the paper's ablation labels as follows:
   `A1_star`→A2, `A_text`→A3, `A2_no_focal`→A4, `A3_no_sampler`→A5,
   `A_Nq16`→A7, `A_Nq4`→A8; the encoder-variant ablations (paper A1 floor
   and A6 linguistic-feature encoder) were run with earlier encoder-search
   tooling and are not part of this release.
3. **Corpus-confound analyses** (after the main experiments have produced
   `results/rl_experiments/`):
   - `python confound_analysis.py --step e2` (per-corpus Control recall)
   - `python confound_analysis.py --step e2prime` (LOCO Control-F1)
   - `python confound_analysis.py --step e3prime` (tabular permutation test)
   - `python confound_analysis.py --step e3w` (WAB-features-fixed evaluation)
   - `python confound_analysis.py --step e1` (corpus-identity probe)
   - `python correlation_analysis.py` (inter-disorder structure figures)
4. **Library modules**: `baselines.py` (`run_all_baselines`),
   `error_analysis.py` (`run_error_analysis`), and `statistical_analysis.py`
   (`mcnemar_test`, `bootstrap_permutation_test`, `aggregate_seeds`) are called
   on the loaded datasets / stored predictions rather than run as scripts.

## Repository layout

```
config.py                  paths, hyperparameters, label maps
data_loader.py             corpus loading, patient-level splits, PyTorch datasets
feature_extractor.py       12 linguistic features
audio_extractor.py         wav2vec2-large segment features (30 s chunks)
paralinguistic_extractor.py 15 acoustic-prosodic features
models/                    encoders, Q-Former, task heads, full model
trainer.py                 training loop (focal loss, weighted sampling)
eval_utils.py              evaluation incl. prior-correction calibration
run_rl_all_experiments.py  experiment definitions and runner
confound_analysis.py       corpus-confound analyses
baselines.py               trivial + traditional-ML baselines
error_analysis.py          per-corpus error analysis
correlation_analysis.py    inter-disorder correlation structure (figures)
statistical_analysis.py    significance testing utilities
```

## Notes

- Outputs default to `<data root>/poc/results/`; set the `OUTPUT_DIR`
  environment variable to redirect feature caches and results elsewhere.

## License

MIT (see `LICENSE`). Please cite the accompanying paper.
The TalkBank corpora are governed by their own usage agreements and are not
covered by this license.
