# 🧬 MD-ContactML

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange)
![MDAnalysis](https://img.shields.io/badge/MDAnalysis-Trajectory%20Processing-brightgreen)
![Status](https://img.shields.io/badge/Status-Active-success)

**MD-ContactML** is a machine learning pipeline for analyzing Molecular Dynamics (MD) trajectories, extracting dynamic residue–residue contact networks at protein–protein interfaces, and performing iterative feature elimination to isolate the most critical, non-redundant biophysical interactions.

---

## 🛠 Tech Stack

- **Data Processing & MD:** [MDAnalysis](https://www.mdanalysis.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/) (Random Forest, MLP Neural Networks, Logistic Regression)
- **Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Utilities:** [tqdm](https://tqdm.github.io/) (Progress Tracking)

---

## 🔬 Overview

In MD simulations of viral complexes (such as SARS-CoV and SARS-CoV-2 Spike RBD complexed with the human ACE2 receptor), thousands of pairwise atomic distances fluctuate over time. Many of these contact distances are highly collinear and redundant.

**MD-ContactML** implements an iterative feature elimination and evaluation loop:
1. **Feature Extraction**: Extracts pairwise $C_\alpha$ interface distances across trajectory frames and replicas using MDAnalysis.
2. **Contact Pruning**: Selects the closest interacting atom pairs representing the active binding interface.
3. **Iterative Elimination**: 
   - Detects the most highly correlated feature pairs ($r \ge 0.90$).
   - Drops the more globally redundant feature.
   - Evaluates machine learning classifiers at each iteration to ensure classification accuracy is preserved.
4. **Visualization**: Generates accuracy decay curves, correlation heatmaps before/after pruning, and final surviving feature interaction maps.

---

## 📁 Project Structure

```text
MD-ContactML/
├── data/
│   └── trajectories/
│       └── final/           # Topology (.parm7) & PDB trajectories (rep1..5)
├── results/
│   ├── figures/             # Visualizations (heatmaps, accuracy curves)
│   ├── elimination_log.csv  # Step-by-step elimination tracking
│   └── final_features.txt   # Surviving non-redundant contact pairs
├── ml_pipeline.py           # Core library (feature extraction, ML models, logic)
├── run_pipeline.py          # Main execution script
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation
Ensure you have Python 3.9+ installed. Install the required dependencies:

```bash
git clone https://github.com/N-S8990/MD-ContactML.git
cd MD-ContactML
pip install -r requirements.txt
```

### 2. Running the Pipeline
To run the full feature extraction and elimination pipeline:

```bash
python run_pipeline.py
```

### 3. Custom Parameters
You can adjust correlation thresholds, accuracy tolerances, and feature limits via CLI arguments:

```bash
python run_pipeline.py \
  --corr_threshold 0.85 \
  --acc_tolerance 0.05 \
  --min_features 20 \
  --out_dir results
```

---

## 🤖 Machine Learning Models

The pipeline evaluates contact networks using three models concurrently:
* **Multi-Layer Perceptron (MLP)**: Feedforward Neural Network (128 & 64 neurons), ReLU activation, Adam optimizer.
* **Random Forest (RF)**: 100 decision estimators for robust ensemble learning.
* **Logistic Regression (LR)**: L-BFGS solver with standard scaling.

---

## 📊 Outputs

* **`results/elimination_log.csv`**: Full step-by-step log of features dropped, correlations, and classifier metrics.
* **`results/final_features.txt`**: List of the surviving non-redundant residue contact pairs.
* **`results/figures/`**: Visualizations including accuracy curves and correlation heatmaps.

---
*Built for computational biophysics and structural bioinformatics research.*