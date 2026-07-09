# Music Genre and Composer Classification Using Deep Learning

## Project Overview
This project uses deep learning to classify classical music composers from MIDI files. The selected composers are Bach, Beethoven, Chopin, and Mozart.

## Objective
The goal is to build and compare two PyTorch models:

1. LSTM model for sequence-based MIDI note classification
2. CNN model for matrix-based music representation classification

## Dataset

The project uses the **MIDI Classic Music** dataset obtained from Kaggle.

Only the following four composers are included in this project:

- Bach
- Beethoven
- Chopin
- Mozart

The dataset is organized as follows:

```
data/
└── raw/
    ├── Bach/
    ├── Beethoven/
    ├── Chopin/
    └── Mozart/
```

The original dataset was cleaned by removing all other composers and unnecessary files so that only the four required classes remain.

The dataset can be downloaded here:

**Google Drive:**  
https://drive.google.com/drive/folders/1m_08SiSKKCfqiFbQVvNn7cIf7ZCiVdGd?usp=sharing


## Methods
The project includes:

- MIDI data loading
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
- music21 or pretty_midi

## Deliverables
- Project Notebook
- Project Report

## Project Structure

- data/ - Dataset
- notebooks/ - Jupyter notebooks
- src/ - Python source code
- models/ - Saved trained models
- reports/ - Final project report

## Team Members

- Marston Ward
- Josue Sandoval
- Saman Tavasoli

