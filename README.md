# PACS: Perturbable Artificial Cognitive System

**English** | [中文](README.zh-CN.md)

PACS turns a large language model into an **artificial mind** through prompt-level instantiation, applies selective perturbations to its basic cognitive computations, and studies how complex behavioural phenotypes emerge — together with autism-like mechanisms, computational subtypes, and individual differences.

```
cognitive primitives → selective perturbation → phenotype emergence → mechanism fitting → falsifiable prediction → human validation
```

This repository is the paper's accompanying release of code, dialogue data, result tables, and figures. It contains:

- a self-contained agent implementation (`pacs/`);
- per-subject dialogue data from five experimental batches (compressed in `pacs/data.7z`; extracts to `pacs/data/runs/`);
- four result tables, four main figures, and their metric manifests (`results/`);
- all scripts needed to recompute the tables and figures from the dialogue data (`results/scripts/`).

> **Intended use.** This is methodological research material. All subjects are **simulated minds** instantiated by an LLM, not data from real children. It must not be used for medical diagnosis or as a substitute for professional assessment.

---

## Contents

- [1. System overview](#1-system-overview)
- [2. Repository layout](#2-repository-layout)
- [3. Quick start](#3-quick-start)
- [4. Connecting a real model](#4-connecting-a-real-model)
- [5. Dialogue dataset](#5-dialogue-dataset)
- [6. Result tables and figures](#6-result-tables-and-figures)
- [7. Reproducing tables and figures](#7-reproducing-tables-and-figures)
- [8. Documentation](#8-documentation)
- [9. Tests](#9-tests)
- [10. Limitations and scope](#10-limitations-and-scope)
- [11. License and citation](#11-license-and-citation)

---

## 1. System overview

### 1.1 Research logic

In real autism research, the underlying cognitive mechanisms cannot be manipulated directly. Researchers can only infer mechanisms from symptom combinations, and the mapping between symptoms and mechanisms is many-to-many. PACS reverses the direction: **construct the mechanism first, then observe the phenotype.**

The system decomposes the mind into three levels:

```
Level 0   31 independently perturbable cognitive primitives (8 families), integrity vector θ = (P₁ … P₃₁)
             │  selective perturbation: 1.0 = intact
             ▼
Level 1   higher-order cognition (theory of mind, cognitive flexibility, pragmatic interpretation)
             │  emerges from combinations of primitives; never perturbed directly
             ▼
Level 2   behavioural phenotype: 6 phenotype dimensions + 1 control channel
```

Operational definitions, theoretical grounding, and computational roles of the 31 primitives are in `pacs/src/pacs/ontology/primitives.py`; the 6 phenotype dimensions and the control channel are in `pacs/src/pacs/phenotype/dims.py`.

### 1.2 Three roles and blinding

| Role | Responsibility | Can see | Cannot see |
|---|---|---|---|
| **Simulated brain** | the artificial mind perturbed by θ | its own integrity parameters, the examiner's turn-by-turn questions | the existence or judgements of the examiner and analyst |
| **Examiner** | another LLM that designs situations and asks questions | observation targets, the reference item bank, the dialogue so far | θ, primitive names, integrity values |
| **Analyst** | another LLM that reads the full dialogue and scores each dimension | the full transcript, scoring anchors, evidence coverage | θ, internal construction |

θ enters only the simulated brain's prompt. The examiner and analyst inputs contain no θ, and the signature of `analyze_dialogue()` has no θ argument — readings cannot be directly influenced by the perturbation parameters at the code level.

### 1.3 Measurement protocol

Every observation follows the same protocol:

| Item | Value |
|---|---|
| Mental age | fixed at **36 months**; no configuration entry |
| Base turns | **35 turns** = 7 scored channels × 5 questions per channel |
| Follow-up | channels still below quota after the base turns receive explicit per-turn instructions to continue probing |
| Turn cap | default **100** (hard cap 500) |
| Question/answer length | default **≤100 characters**; over-length is warned about, never truncated |
| Analyst output | per-dimension readings (0–100; 100 = age-typical) plus a ~350-character case-report-style conclusion |

One dialogue turn = 2 model calls (question + answer), plus 1 analyst call at the end. The 35-turn base is about 71 calls per subject; follow-up to the 100-turn cap is at most about 201 calls.

### 1.4 Construct-validity constraints

- **There is no predefined mechanism → measurement-dimension mapping.** Which dimension is affected is determined by the data: the dimension with the largest drop among the six phenotype dimensions is treated as dominant, then effect concentration and control-channel stability are checked.
- **No evidence ≠ typical performance.** Channels that were never probed receive empty readings and are listed in `uncovered_channels`; they are never filled with typical values.
- **Synthetic data must not masquerade as emergence.** Offline mock profiles are marked `provider_kind="mock"` / `synthetic=true`, and reports carry a disclaimer.
- These constraints are enforced by AST scanning, prompt-blindness checks, and synthetic-data guards in the test suite.

---

## 2. Repository layout

```
PACS-publish/
├── README.md                        English README (default)
├── README.zh-CN.md                  Chinese README
├── LICENSE                          MIT
├── environment.yml                  conda environment (tables and figures)
├── requirements.txt                 pip dependencies (tables and figures)
│
├── pacs/                            agent and experiment drivers
│   ├── src/pacs/                    core package (zero third-party runtime dependencies)
│   ├── tests/                       399 tests
│   ├── tools/                       experiment drivers and item-bank builder
│   ├── data.7z                      five batches of per-subject dialogue data (extracts to data/runs/)
│   ├── pyproject.toml
│   ├── config.example.json          configuration template
│   ├── README.md                    technical documentation (Chinese)
│   └── tools/可靠性实验.md            Fig.1 reliability design and metrics (Chinese)
│
└── results/                         result tables, figures, and reproduction scripts
    ├── scripts/                     9 scripts
    ├── data/                        four result tables, candidate-link tables, dialogue casebook
    └── figure/                      Fig.1–4 png/pdf, metrics manifests
```

The release is about **31 MB**, of which **21 MB** is the `pacs/data.7z` dialogue archive; extracting it produces `pacs/data/runs/` (about 108 MB, not tracked by Git).

---

## 3. Quick start

The agent requires only Python ≥ 3.10 and has **zero third-party runtime dependencies**. The offline mock mind is enough to run the full pipeline:

```bash
cd pacs
set PYTHONPATH=src            # Windows cmd
                              # PowerShell: $env:PYTHONPATH="src"
                              # Linux / macOS: export PYTHONPATH=src

python -m pacs.cli --version          # pacs 0.1.0
python -m pacs.cli ontology           # self-check of the 31 primitives and probe registry
python -m pacs.cli demo               # offline end-to-end demo (mock mind; no API key)
python -m pacs.cli serve              # visual interface (default 127.0.0.1)
```

Run the test suite:

```bash
cd pacs
set PYTHONPATH=src
python -m pytest tests
```

Current suite: **399 tests** (398 pass; 1 skips automatically when the external AsdKB source data are unavailable).

---

## 4. Connecting a real model

Copy the configuration template and fill in provider, model, base URL, and API key:

```bash
cd pacs
copy config.example.json config.json       # Windows
# cp config.example.json config.json       # Linux / macOS
```

Environment variables can override the file without writing secrets to disk:

| Variable | Purpose |
|---|---|
| `PACS_LLM_PROVIDER` / `PACS_LLM_API_KEY` / `PACS_LLM_MODEL` / `PACS_LLM_BASE_URL` | simulated brain |
| `PACS_EXAMINER_LLM_*` | examiner / analyst (inherits the simulated-brain settings when unset) |

Any OpenAI-compatible endpoint works (DeepSeek, OpenAI, local vLLM / Ollama, and so on).

```bash
# Initialise an intact mind and inspect its self-report
python -m pacs.cli init-brain --provider openai --model gpt-4o-mini

# Print the mind prompt for one perturbation
python -m pacs.cli prompt --primitive social_reward_weighting --dose 0.2

# Preview design size and call count (no model calls)
python -m pacs.cli sample --mode all --limit 30

# Small real run
python -m pacs.cli run --provider openai --model gpt-4o-mini --mode single --limit 3
```

**Never** commit a `config.json` containing an API key.

---

## 5. Dialogue dataset

Dialogue data are distributed as `pacs/data.7z`. Extracting it produces `pacs/data/runs/`. Each batch contains:

```
<run>/
├── cells/<condition>/subjects/<index>_theta_<id>_r<rep>.json   per subject: parameters + turns + readings + profile
├── flat/all_subjects.csv                                       wide table, one row per subject
├── analysis/                                                   batch-level aggregates
├── manifest.json                                               run parameters, models, protocol, cost
└── summary.md                                                  human-readable summary
```

| Batch | Design | Subjects | Used for |
|---|---|---|---|
| `fig1_main` | 7 conditions × 4 θ levels, 3–5 replicates | 116 | `result_1`, Fig.1 |
| `fig1_realcheck` | supplementary reference arm at the same θ (5 intact minds + 5 perturbed) | 10 | behavioural reference for dialogue evidence |
| `scale_single` | 31 primitives × 2 levels × 3 replicates + 3 generic-degradation controls | 189 | `result_2`, Fig.2, Fig.3 |
| `scale_pair` | C(31,2) = 465 pairs × 3 replicates | 1395 | `result_3`, Fig.4 |
| `scale_single_dose_merged_5dose_5rep` | 5 primitives × 5 doses × 5 replicates | 125 | `result_4`, Fig.2 panel F, Fig.3 Q3 |

Each per-subject JSON has five blocks:

| Block | Content |
|---|---|
| `parameters` | θ, generic degradation, design, origin, θ fingerprint |
| `run_parameters` | turns, cap, text limit, seed, probe mode |
| `dialogue` | session metadata and **turn-by-turn** `{turn, role, content, focus}` |
| `analysis` | seven channel readings, per-dimension detail, analyst conclusion, ASD-likeness impression |
| `profile` | phenotype profile, provider kind, `synthetic` flag, uncovered channels |

Extraction (from `pacs/`):

```bash
7z x data.7z
```

On Windows you can also right-click `data.7z` and choose “Extract Here”. After extraction, each batch's `flat/数据字典.md` documents every field.

---

## 6. Result tables and figures

### 6.1 Result tables

| File | Content | Source batch |
|---|---|---|
| `results/data/result_1_robustness.xlsx` | 116 subjects × 16 columns; four-axis reliability experiment | `fig1_main` |
| `results/data/result_2_univariate.xlsx` | 189 subjects × 13 columns; single-primitive perturbation batch | `scale_single` |
| `results/data/result_3_multivariable.xlsx` | 1395 subjects; pairwise combination batch | `scale_pair` |
| `results/data/result_4_does_response.xlsx` | 125 subjects; five-dose ladder | `scale_single_dose_merged_5dose_5rep` |

The first sheet in each workbook, `by_subject`, is one row per subject; later sheets contain cell-level aggregates, dose–response fits, and specificity verdicts.

### 6.2 Figures

| Figure | File | Content |
|---|---|---|
| **Fig.1** | `results/figure/figure1_pacs_reliability.{png,pdf}` | system reliability: same-brain retest, model swaps, examiner/analyst/order swaps, turn-count changes |
| **Fig.2** | `results/figure/figure2_univariate_phenotypes.{png,pdf}` | single-primitive batch: phenotype heat maps, Manhattan plots, bipartite networks, dose–response |
| **Fig.3** | `results/figure/figure3_dialogue_evidence.{png,pdf}` | dialogue evidence: impaired vs intact minds, same dimension/different mechanisms, same mechanism/different doses |
| **Fig.4** | `results/figure/figure4_interaction_stability.{png,pdf}` | stability of combination effects: synergy, buffering, new phenotypes, replicate agreement |

Each figure has a companion `figure*_metrics.json` containing every number shown, panel definitions, and the fields needed for reproduction.

The complete dialogue casebook is at `results/data/dialogue_evidence_casebook.{md,xlsx}`; the candidate-link tables are at `results/data/figure2_link_candidates.{csv,md}`.

---

## 7. Reproducing tables and figures

Environment:

```bash
conda env create -f environment.yml && conda activate pacs
# or: python -m pip install -r requirements.txt
```

Scripts in `results/scripts/` resolve the project root from their own location and add `pacs/src` and `pacs/tools` to `sys.path`; **no `PYTHONPATH` is needed**.

### 7.1 Result tables

Extract the dialogue data once:

```bash
cd pacs
7z x data.7z
cd ..
```

```bash
cd results/scripts

# result_2 (full 16-column export) and result_3
python export_result_tables.py

# result_4 (five-dose ladder)
python export_dose_supplement.py \
  --merged-run pacs/data/runs/scale_single_dose_merged_5dose_5rep \
  --doses 0.2,0.4,0.6,0.8,0.9 --replicates 5 \
  --by-subject-layout result2 --no-ladder --freeze-panes A44 \
  --xlsx-name result_4_does_response.xlsx --out-dir ../data
```

The released `result_2_univariate.xlsx` has **13 columns** (the three model columns are omitted); `export_result_tables.py` writes **16 columns** by default. Row order and all readings are identical. `result_1_robustness.xlsx` is produced by the upstream Fig.1 pipeline and is shipped as-is; it does not need to be rerun.

### 7.2 Figures

Run in the following order — later scripts depend on earlier outputs:

```bash
cd results/scripts

# Fig.1: reliability → figure1_* + figure1_metrics.json
python fig1_visualize.py

# Fig.2: single-primitive phenotypes → figure2_* + figure2_metrics.json
python fig2_visualize.py

# Fig.2 candidate-link tables → figure2_link_candidates.{csv,md}
python fig2_link_candidates.py

# dialogue casebook → dialogue_evidence_casebook.{md,xlsx} + dialogue_evidence_metrics.json
python analyze_dialogue_evidence.py

# Fig.3: dialogue evidence → figure3_* + figure3_metrics.json
python fig3_dialogue_evidence.py

# Fig.4: combination-effect stability → figure4_* + figure4_metrics.json
python fig4_interaction_stability.py
```

Every script runs a consistency gate before writing: for example, Fig.2 reconciles all 62 verdicts row by row, Fig.3 recomputes 186 p-values, and Fig.4 checks ten chain-consistency gates. If a gate fails, no figure is written.

---

## 8. Documentation

| Document | Content |
|---|---|
| [`pacs/README.md`](pacs/README.md) | technical documentation (Chinese): installation, CLI, HTTP routes, configuration, storage, tests |
| [`pacs/tools/可靠性实验.md`](pacs/tools/可靠性实验.md) | Fig.1 reliability experiment: four axes, statistical metrics, reporting conventions (Chinese) |

---

## 9. Tests

```bash
cd pacs
set PYTHONPATH=src
python -m pytest tests
```

Coverage includes construct-validity guards (no predefined mappings), prompt blindness, turn quotas and text limits, analyst-output parsing, the freeze protocol, HTTP API, web interface, and the item-bank build chain. Current suite: **399 tests**, 398 pass, 1 skipped when the external AsdKB source data are unavailable.

---

## 10. Limitations and scope

1. **All subjects are simulated minds.** Any conclusion about autism mechanisms must come from runs with a real model; the mock path validates only the pipeline and statistical logic.
2. **Stability of prompt-level perturbation is the central untested assumption.** If a primitive does not produce stable, specific effects in a real LLM, it should be removed according to the pre-registered criteria.
3. **Phenotype resolution is limited by analyst stability.** Per-dimension readings come from an LLM's interpretation of one dialogue; dimensions within a session are correlated, and only across sessions are they independent. Increase independent replicates rather than adding more questions.
4. **This work does not diagnose.** Mechanism fitting answers “which computational mechanism does this mind fit better?”, not `LLM score → ASD/TD`.
5. **Human validation is outside this repository.** The repository provides falsifiable, frozen predictions and the code needed to reproduce them.
6. **Item-level mapping in the examiner bank is derived by this project.** The AsdKB per-item symptom associations are effectively scale-level coverage, so dimension/symptom/criterion links are derived by explicit rules and annotated item by item.

> Simulation supplies priors and mechanistic hypotheses, not truth; real human data are responsible for correcting the priors.

---

## 11. License and citation

Code and data are released under the [MIT](LICENSE) license.

If you use this repository, please cite:

```bibtex
@software{pacs2026,
  title  = {PACS: Perturbable Artificial Cognitive System},
  year   = {2026},
  note   = {Research code and data release},
  url    = {https://github.com/hcji/PACS}
}
```

> **Intended use.** This project is methodological research material. All subjects are simulated. It is **not for medical diagnosis** and is not a substitute for professional assessment.
