# Music Composer Classification Using Deep Learning

## Project Overview
This project uses deep learning to classify classical piano works by composer from symbolic MIDI files. The prediction task is restricted to four composers: **Bach, Beethoven, Chopin, and Mozart**.

## Objective
The goal is to build and compare two PyTorch models for four-way composer classification:

1. LSTM model for sequence-based MIDI note classification
2. One-dimensional CNN model for local symbolic-note pattern classification

## Dataset

The dataset for this project is the **MIDI Classic Music** collection from Kaggle (Blanderbuss, 2025):  
<https://www.kaggle.com/datasets/blanderbuss/midi-classic-music>

The full archive contains works by many composers. This project uses only the four required composers:

- **Bach**
- **Beethoven**
- **Chopin**
- **Mozart**

The helper script [`src/prepare_dataset.py`](src/prepare_dataset.py) downloads
the archive. The master workflow then removes unreadable files and exact
duplicates before creating its deterministic 70/15/15 train, validation, and
test manifest.

### Class distribution after filtering

| Composer  | Train | Validation | Test | Total |
|-----------|-------|-----|------|-------|
| Bach      | 709   | 152 | 153  | 1014  |
| Beethoven | 146   | 31  | 31   | 208   |
| Chopin    | 92    | 20  | 20   | 132   |
| Mozart    | 178   | 38   | 38   | 254   |

The 1,608-file clean manifest excludes 22 invalid or duplicate MIDI files. The
training split is **imbalanced** (Bach is ~7.7× larger than Chopin), so both
baselines use inverse-frequency class-weighted cross-entropy loss.

> **Do not push the dataset to GitHub.** The raw MIDI files are large and should not be committed to version control. `dataset/` is already ignored by git.

## EDA Findings

Exploratory data analysis (EDA) is run and cached by
[`notebooks/master_composer_classification.ipynb`](notebooks/master_composer_classification.ipynb).

### Parsing and data quality

- Clean manifest: **1,608 valid files**
- Exclusions: **22 malformed or duplicate files**
- EDA metadata and plots are generated once, then cached for subsequent runs.

### Key quantitative findings (train split)

- **Class imbalance is substantial**: 709 Bach files vs 92 Chopin files.
- **Duration differs strongly by composer**:
	- Bach: 160.3s mean
	- Beethoven: 518.3s mean
	- Chopin: 222.3s mean
	- Mozart: 415.5s mean
- **Note density is discriminative**:
	- Bach: 9.3 notes/s
	- Beethoven: 14.5 notes/s
	- Chopin: 11.6 notes/s
	- Mozart: 13.7 notes/s
- **Pitch range separates styles/eras**:
	- Bach: 43.1 semitones
	- Beethoven: 61.1 semitones
	- Chopin: 63.0 semitones
	- Mozart: 55.3 semitones

### Feature-space insight

- The master workflow retains note sequences rather than reducing the task to
	summary features, allowing both baselines to model pitch and timing patterns.
- Standardized PCA explains 47.1% and 18.5% of variance in its first two
	components (65.6% cumulative), but composer classes still overlap heavily.
- Note count correlates with duration ($r = .69$), supporting the use of note
	density and sequence timing rather than raw counts alone.

### Modeling decisions informed by EDA

- Use **class-weighted loss** for imbalance.
- Track **per-class precision/recall** and confusion matrices, not just overall accuracy.
- Standardize fixed-length note-sequence representations for both LSTM and CNN pipelines.

## Current Validation Results

The master notebook was executed on the fixed clean manifest using PyTorch MPS
on Apple Silicon. The held-out test split was not used.

Early runs without regularization showed clear overfitting: training accuracy
kept climbing past 90% (CNN: 96.8%) while validation loss stopped improving and
became unstable. Both baselines now use Adam with weight decay (1e-4), retain
the best validation checkpoint, and apply a five-epoch early-stopping criterion.

| Model | Best validation epoch | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|---:|
| LSTM | 6 | 0.759 | 0.748 | 0.622 |
| CNN | 18 | **0.822** | **0.812** | **0.710** |

In this seeded MPS run, the CNN improved validation accuracy by 6.3 percentage
points and weighted F1 by 6.4 points. A separate evaluation of the selected CNN
checkpoint measured 98.2% training accuracy versus 82.2% validation accuracy,
so substantial overfitting remains despite regularization and checkpoint
selection. Accelerator kernels can produce run-to-run variation even with fixed
data splits and random seeds. Generated EDA, loss, and confusion-matrix figures
are tracked in [`figures/`](figures).

## Methods
The project includes:

- MIDI data loading (via Kaggle download)
- Data filtering and train/validation/test splitting
- Data preprocessing
- Feature extraction
- Label encoding
- PyTorch Dataset and DataLoader creation
- LSTM model training
- CNN model training
- Model evaluation using accuracy, precision, recall, and confusion matrix

## Tools
- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- pretty_midi
- Kaggle API


## Setup
1. Clone the repository.

2. Install [Pixi](https://pixi.sh/) if it is not already available on your system.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

3. Create the project-local environment from the tracked lockfile:

```bash
pixi install
```

Pixi creates `.pixi/` inside this repository. It is local to each collaborator
and is not committed; `pixi.toml` and `pixi.lock` define the shared environment.

4. Start JupyterLab through Pixi:

```bash
pixi run notebook
```

5. Select the Pixi Python environment as the VS Code notebook kernel, then run
the master notebook in order:

```bash
notebooks/master_composer_classification.ipynb
```

On Apple Silicon, PyTorch automatically uses the Metal Performance Shaders (MPS)
backend when it is available. `src/training.py`'s `choose_device()` prefers MPS,
then falls back to CUDA on Linux/Windows machines with an NVIDIA GPU, and finally
CPU. `pixi.toml` locks `osx-arm64`, `linux-64`, and `win-64`, so any teammate can
run `pixi install` on their own machine and train with whichever accelerator is
available.

### Useful Pixi Commands

```bash
pixi run check-imports
pixi run prepare-dataset
pixi run notebook
pixi run test
```

To execute the master noninteractively after preparation:

```bash
pixi run jupyter nbconvert --to notebook --execute --inplace \
	--ExecutePreprocessor.timeout=0 notebooks/master_composer_classification.ipynb
```

## Deliverables
- Project Notebook
- Project Report

## Project Structure

- `.pixi/` — Project-local Pixi environment (not tracked by Git)
- `pixi.toml` / `pixi.lock` — Shared, reproducible environment definition
- `dataset/` — Raw MIDI datasets and local processing caches (gitignored)
- `figures/` — Reproducible PNG plots exported from the executed master notebook
- `notebooks/` — Jupyter notebooks
- `src/` — Python source code
- `models/` — Saved trained models
- `report/` — APA7 LaTeX final project report (`report/main.tex`)


## Team Members

- Marston Ward
- Josue Sandoval
- Saman Tavasoli
