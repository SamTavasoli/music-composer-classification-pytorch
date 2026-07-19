# Music Composer Classification Using Deep Learning

## Project Overview
This project uses deep learning to classify classical piano works by composer from symbolic MIDI files. The prediction task is restricted to four composers: **Bach, Beethoven, Chopin, and Mozart**.

## Objective
The goal is to build and compare two PyTorch models for four-way composer classification:

1. LSTM model for sequence-based MIDI note classification
2. CNN model for matrix-based music representation classification

## Dataset

The dataset for this project is the **MIDI Classic Music** collection from Kaggle (Blanderbuss, 2025):  
<https://www.kaggle.com/datasets/blanderbuss/midi-classic-music>

The full archive contains works by many composers. This project uses only the four required composers:

- **Bach**
- **Beethoven**
- **Chopin**
- **Mozart**

The helper script [`src/prepare_dataset.py`](src/prepare_dataset.py) downloads the archive from Kaggle, filters these four composers, and creates reproducible `train`, `dev`, and `test` splits.

### Class distribution after filtering

| Composer  | Train | Dev | Test | Total |
|-----------|-------|-----|------|-------|
| Bach      | 709   | 152 | 153  | 1014  |
| Beethoven | 146   | 31  | 31   | 208   |
| Chopin    | 92    | 20  | 20   | 132   |
| Mozart    | 178   | 38  | 38   | 254   |

The filtered dataset is **imbalanced** (Bach is ~7.5× larger than Chopin). We address this with class-weighted cross-entropy loss and pitch-shift / time-stretch data augmentation for the minority classes.

> **Do not push the dataset to GitHub.** The raw MIDI files are large and should not be committed to version control. `dataset/` is already ignored by git.

## EDA Findings

Exploratory data analysis (EDA) was run on the filtered four-composer dataset in `notebooks/eda.ipynb`.

### Parsing and data quality

- Successfully parsed files: **1,628 / 1,630**
- Parse failures: **2 files** (unsupported key-signature metadata)
- All key EDA plots are exported to `figures/`.

### Key quantitative findings (train split)

- **Class imbalance is substantial**: 717 Bach files vs 95 Chopin files.
- **Duration differs strongly by composer**:
	- Bach: 160.0s mean
	- Beethoven: 529.0s mean
	- Chopin: 223.8s mean
	- Mozart: 393.5s mean
- **Note density is discriminative**:
	- Bach: 9.5 notes/s
	- Beethoven: 14.0 notes/s
	- Chopin: 11.3 notes/s
	- Mozart: 13.9 notes/s
- **Pitch range separates styles/eras**:
	- Bach: 42.8 semitones
	- Beethoven: 62.1 semitones
	- Chopin: 61.9 semitones
	- Mozart: 55.3 semitones

### Feature-space insight

- PCA on engineered features shows only **modest separation** between classes.
- First two PCs explain approximately **54.8%** of variance (PC1: 40.1%, PC2: 14.7%).
- Conclusion: summary statistics alone are not sufficient; sequence-aware inputs (LSTM note sequences and CNN piano rolls) are justified.

### Modeling decisions informed by EDA

- Use **class-weighted loss** for imbalance.
- Apply **augmentation** to minority composers.
- Track **per-class precision/recall** and confusion matrices, not just overall accuracy.
- Standardize fixed-size representations for both LSTM and CNN pipelines.

## Methods
The project includes:

- MIDI data loading (via Kaggle download)
- Data filtering and train/dev/test splitting
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

2. Create a Python virtual environment:

```bash
python -m venv .venv
```

3. Activate the virtual environment.

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

4. Install the required Python packages:

```bash
pip install -r requirements.txt
```

5. Open the project in Visual Studio Code (or your preferred IDE).

6. Select the project's **.venv** as the Python interpreter or Jupyter Notebook kernel before running the notebooks.

## Deliverables
- Project Notebook
- Project Report

## Project Structure

- `.venv/` — Project Python virtual environment (not tracked by Git)
- `dataset/` — Raw and filtered MIDI datasets (gitignored)
- `figures/` — Generated plots
- `notebooks/` — Jupyter notebooks
- `src/` — Python source code
- `models/` — Saved trained models
- `report/` — APA7 LaTeX final project report (`report/main.tex`)


## Team Members

- Marston Ward
- Josue Sandoval
- Saman Tavasoli
