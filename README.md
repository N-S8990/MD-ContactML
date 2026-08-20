# MD-ContactML

**MD-ContactML** is a machine learning pipeline for analyzing Molecular Dynamics (MD) trajectories, extracting dynamic residue–residue contact networks at protein–protein interfaces, and performing iterative feature elimination to isolate the most critical, non-redundant biophysical interactions.

---

## 🧬 Overview

In MD simulations of viral complexes (such as SARS-CoV and SARS-CoV-2 Spike RBD complexed with the human ACE2 receptor), thousands of pairwise atomic distances fluctuate over time. Many of these contact distances are highly collinear and redundant.

**MD-ContactML** implements an iterative feature elimination and evaluation loop:
1. **Feature Extraction**: Extracts pairwise $C_\alpha$ interface distances across trajectory frames and replicas using MDAnalysis.
2. **Contact Pruning**: Selects the closest interacting atom pairs representing the active binding interface.
3. **Iterative Elimination**: 
   - Detects the most highly correlated feature pairs ($r \ge 0.90$).
   - Drops the more globally redundant feature.
   - Evaluates machine learning classifiers (Random Forest, Multi-Layer Perceptron Neural Network, Logistic Regression) at each iteration to ensure classification accuracy is preserved.
4. **Visualization**: Generates accuracy decay curves, correlation heatmaps before/after pruning, and final surviving feature interaction maps.

---

## 📁 Project Structure

```
MD-ContactML/
├── data/
│   └── trajectories/
│       └── final/
│           ├── sars-cov-2/          # Topology (.parm7) & PDB trajectories (rep1..5)
│           ├── sars-cov-2002/       # Topology (.parm7) & PDB trajectories (rep1..5)
│           ├── sars-cov-2-no-ace2/
│           └── sars-cov-2002-no-ace2/
├── results/
│   ├── figures/
│   │   ├── accuracy_vs_features.png
│   │   ├── final_features.png
│   │   └── heatmap_after.png
│   ├── elimination_log.csv
│   └── final_features.txt
├── ml_pipeline.py                   # Core library (feature extraction, ML models, elimination loop)
├── run_pipeline.py                  # Main execution script
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation
Ensure you have Python 3.9+ installed. Install the required dependencies:

```bash
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

* **Multi-Layer Perceptron (MLP)**: Feedforward Neural Network with 2 hidden layers (128, 64 neurons), ReLU activation, Adam optimizer.
* **Random Forest (RF)**: 100 estimators.
* **Logistic Regression (LR)**: L-BFGS solver with standard scaling.

---

## 📊 Outputs

* **`results/elimination_log.csv`**: Full step-by-step log of features dropped, correlations, and classifier metrics.
* **`results/final_features.txt`**: List of the surviving non-redundant residue contact pairs.
* **`results/figures/`**: Visualizations including accuracy curves and correlation heatmaps.
