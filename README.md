# XAI-Driven Adaptive Trust Scoring for a Secure DNS Authentication Framework

Extension of the SATAF / D-TS thesis. A calibrated, explainable trust score
gates entry into the deterministic DANE/TLSA validation path — it does not
replace it.

## Architecture principle

    probabilistic perimeter  ->  deterministic core
    (ML/DL trust score)          (RRSIG verification, TLSA hash match)

ML decides *what to be suspicious of*. Cryptography decides *what is valid*.
Bolting a classifier onto signature validation is AI-washing; security venues
punish it. Nothing in `src/models/` ever overrides a cryptographic verdict.

## Where things live

Code is version-controlled here. Data and outputs live on Google Drive and are
never committed.

    GitHub (this repo)          Google Drive
    ------------------          ------------
    src/                        data/00_raw/         WRITE-ONCE  public downloads
    notebooks/                  data/01_collected/   WRITE-ONCE  our own scans
    configs/                    data/02_interim/
                                data/03_features/
                                data/04_splits/      frozen + hashed
                                artifacts/{models,predictions,shap_values,checkpoints,logs}/
                                results/{tables,figures}/
                                run_manifest.jsonl
                                _LEAKAGE_NOTES.md

`configs/paths.yaml` is the only place a storage location is defined.

## run_id

`{family}_{split}_s{seed}_{counter}` — e.g. `fusion_family_disjoint_v1_s42_0004`.
It is the filename of the model, the predictions, the SHAP values, and the log,
and the key in `run_manifest.jsonl`. One id resolves the entire provenance of
any number in the paper.

## Two storage facts that shape the design

Colab Pro+ provides 2 TB of **Drive**, not RAM; high-RAM runtimes cap near
51 GB (~83 GB on A100). Training data is therefore staged to `/content` and
read from local SSD — `src/utils/io.stage_local()`.

Drive's FUSE layer does not implement POSIX advisory locking correctly, and
SQLite depends on it. The collection ledger therefore lives at
`/content/certificate_ledger.db` with periodic backups to Drive via SQLite's
own backup API. A ledger opened directly on the Drive mount can corrupt on
disconnect — which is exactly when it will happen.

## Order of work

Build and start `02_certificate_collection` **first**, then develop everything
else while it runs. It is the only step whose duration is not under our
control, and how much certificate coverage is actually achieved determines
every downstream decision.

    02 collection (running, resumable, days)
         ├─ 01 data audit + leakage screen
         ├─ 04 split creation
         ├─ 03 feature engineering
         └─ 05/06 baselines and XGBoost on lexical+DNS only
    then 07 CNN-BiLSTM -> 08 fusion -> 09 calibration -> 10 XAI -> 11 figures

## Non-negotiables

- Logic lives in `src/`. Notebooks orchestrate and plot, nothing else.
- `00_raw/` and `01_collected/` are never written after creation. A certificate
  is a point-in-time observation; it cannot be regenerated.
- Splits are files, hashed at creation. `load_split()` refuses a file that has
  changed since — a silently shifted split makes results incomparable.
- Leakage audit runs before the first model. Decisions recorded in
  `_LEAKAGE_NOTES.md`.
- Per-domain predictions are always saved. A reviewer asking for a new metric
  should cost one minute, not one retraining run.
- Report family-disjoint and temporal splits alongside random. The optimism gap
  is a finding, not an embarrassment.
- Headline metrics are PR-AUC and FPR@95%TPR, not accuracy. At ~30:1 imbalance
  accuracy is meaningless, and a false positive blocks a legitimate business.
- Imbalance is handled with focal loss / class weights, never SMOTE:
  interpolating between character embeddings synthesises domains that cannot
  exist.
