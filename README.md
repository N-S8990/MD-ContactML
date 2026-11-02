# 🧬 MD-ContactML

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenMM](https://img.shields.io/badge/OpenMM-MD%20Simulation-3776AB?logo=python&logoColor=white)
![PyMOL](https://img.shields.io/badge/PyMOL-3D%20Rendering-009688)
![MDAnalysis](https://img.shields.io/badge/MDAnalysis-Trajectory%20Processing-brightgreen)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)

**MD-ContactML** is an end-to-end computational biophysics and machine learning framework for simulating macromolecular complexes, extracting dynamic residue–residue contact networks at protein–protein interfaces, eliminating multicollinear features, benchmarking machine learning classifiers, and discovering mechanistic stability hotspots via empirical **Leave-One-Feature-Out (LOFO) ablation**.

The benchmark dataset studies thermal perturbation across physiological (300 K) and elevated (400 K) ensembles of the Human Growth Hormone–Receptor complex (PDB: `1A22`).

---

## 🛠 Tech Stack

- **Molecular Dynamics & Structure Prep:** [OpenMM](https://openmm.org/), [PDBFixer](https://github.com/openmm/pdbfixer) (Amber14 forcefield, GBn2 Implicit Solvent, Langevin dynamics)
- **Data Processing & MD Trajectories:** [MDAnalysis](https://www.mdanalysis.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [SciPy](https://scipy.org/)
- **Network Graph Theory:** [NetworkX](https://networkx.org/) (Dynamic contact networks, Laplacian spectral connectivity, centrality metrics)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/) (Random Forest, Multi-Layer Perceptron Neural Networks, Logistic Regression, Permutation Importance)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Utilities:** [tqdm](https://tqdm.github.io/) (Progress Tracking)

---

## 🔬 Pipeline Workflow

```
┌────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│  Structure Prep & MD   │ ──> │   Feature Extraction   │ ──> │  Iterative Elimination │
│ (OpenMM / PDBFixer)    │     │  (981 Multi-Modal Feat)│     │   (Threshold = 0.98)   │
└────────────────────────┘     └────────────────────────┘     └────────────────────────┘
            │                                                              │
            ▼                                                              ▼
┌────────────────────────┐                                    ┌────────────────────────┐
│  Zero-Leakage Split    │                                    │  LOFO Empirical Ablat. │
│(Train: Rep1-2 |Test: 3)│                                    │  (Hotspot Discovery)   │
└────────────────────────┘                                    └────────────────────────┘
            │                                                              │
            ▼                                                              ▼
┌────────────────────────┐                                    ┌────────────────────────┐
│  Model Benchmarking    │ ─────────────────────────────────> │  Publication Artifacts │
│  (RF, MLP, Logistic)   │                                    │ (Curves, Heatmaps, CSV)│
└────────────────────────┘                                    └────────────────────────┘
```

1. **Structure Preparation & MD Simulation**: Automated cleaning, missing atom/hydrogen addition, energy minimization, gentle warming, and production MD simulations via OpenMM.
2. **Multi-Modal Biophysical Feature Extraction (981 features)**:
   - **Backbone ($C_\alpha - C_\alpha$) & Sidechain Orientation ($C_\beta - C_\beta$) Distances**.
   - **Minimum Heavy-Atom Contact Distances ($d_{\min}$)** & sidechain centroid (COM) packing vectors.
   - **Continuous Soft Contact Switching Functions**: $s(r) = (1 - (r/5)^6) / (1 - (r/5)^{12}) \in [0, 1]$.
   - **Inverse Proximities**: $1 / (d_{\min} + 0.01)$.
   - **Cation–$\pi$ Interactions**: Direct geometric tracking and continuous contact scoring between basic residues (Arg, Lys) and aromatic rings (Trp, Phe, Tyr, His).
   - **Directional Hydrogen Bonds**: Decomposed into Backbone–Backbone, Sidechain–Sidechain, and Sidechain–Backbone pairs at $3.5\text{ \AA}$ and $3.0\text{ \AA}$ thresholds, plus specific donor–acceptor pairs.
   - **Molecular Dipole Alignment**: Dipole vector norms ($\|\boldsymbol{\mu}_A\|$, $\|\boldsymbol{\mu}_B\|$), alignment angle ($\cos\theta$), and cross-partner charge centroid distances.
   - **Interface Packing Density**: Inter-chain vs. intra-chain interface contact counts and packing ratio.
   - **Salt Bridges & Aromatic $\pi$-Stacking**: Explicit ion pairs and aromatic ring centroid proximities.
   - **Dynamic Network Graph Topology**: Average/max degree, betweenness centrality, degree centrality, clustering coefficient, transitivity, and Laplacian algebraic connectivity ($\lambda_2$).
   - **Backbone & Sidechain Dihedrals**: $\sin$ and $\cos$ components for backbone $\phi, \psi$ and sidechain rotamer $\chi_1$ on both Partner A and Partner B.
   - **Global Shape & Dynamics**: Radii of gyration ($R_g$), shape asphericity, inter-chain COM distance, and backbone RMSD.
3. **Zero-Data-Leakage Replica Splitting**:
   - Replicas 1 and 2 are used for training and inner validation.
   - Replica 3 is strictly held-out as an independent test set.
   - Standard scaling is fitted strictly on training frames.
4. **Iterative Collinearity Elimination Loop**:
   - Computes full Pearson correlation matrix ($|r|$).
   - Iteratively detects and prunes collinear pairs ($|r| \ge 0.98$) in batches.
   - Gated by inner validation accuracy tolerance ($5\%$), stopping if pruning degrades performance.
5. **Multi-Model Benchmark**: Evaluates Random Forest, Logistic Regression, and MLP on the held-out test replica.
6. **Multi-Scale Empirical Ablation (LOFO & Group Cluster Ablation)**:
   - **Single-Feature LOFO**: Systematically drops one feature at a time across all features, retrains, and measures test accuracy drop ($\Delta\text{Acc}$) and log-loss surge.
   - **Group-Level Cluster Ablation**: Simultaneously ablates all mathematical representations belonging to a specific residue–residue contact pair (e.g. $C_\alpha$, $C_\beta$, $d_{\min}$, soft contact, inverse contact, H-bonds) or functional biophysical subsystem (e.g. `CLUSTER_GLOBAL_SHAPE`, `CLUSTER_GRAPH_TOPOLOGY`, `CLUSTER_INTERFACE_PACKING`). This eliminates "redundancy masking" and isolates the true collective causal necessity of each biophysical element.

---

## 📁 Project Structure

```
MD-ContactML/
├── md_simulation.py        # OpenMM MD simulation and structure preparation script
├── ml_pipeline.py          # Core library (feature extraction, ML models, elimination, LOFO, plotting)
├── run_pipeline.py         # Single unified CLI driver running the full workflow end-to-end
├── README.md               # Complete project documentation
├── requirements.txt        # Pinned pip dependencies
├── environment.yml         # Conda environment definition
├── 1A22.pdb                # Reference Growth Hormone - Receptor complex structure
├── data/                   # Simulation trajectories (300 K & 400 K, Replicas 1-3)
└── results/                # Central output directory (reports, CSV data, figures)
    ├── pipeline_report.txt               # Plaintext executive summary report
    ├── raw_features.csv                  # Full feature matrix (1200 frames x 981 features)
    ├── full_correlation_matrix.csv       # Complete Pearson correlation matrix
    ├── elimination_log.csv               # Collinearity elimination logs
    ├── model_benchmark_metrics.csv       # RF, Logistic Regression, MLP metrics
    ├── prominent_features.csv            # Features ranked by RF + Permutation importance
    ├── lofo_feature_importance.csv       # Single features ranked by empirical ablation accuracy drop
    ├── cluster_ablation_importance.csv   # Biophysical clusters ranked by collective ablation accuracy drop
    ├── correlation_matrices/
    │   ├── corr_matrix_initial.csv       # Initial correlation before pruning
    │   └── corr_matrix_final.csv         # Final correlation after pruning
    └── figures/
        ├── full_correlation_heatmap.png       # Complete 981-feature correlation heatmap
        ├── corr_matrix_initial.png            # Initial correlation matrix heatmap
        ├── corr_matrix_final.png              # Collinearity-free correlation matrix heatmap
        ├── elimination_trajectory.png         # Feature count vs. validation accuracy curve
        ├── top_prominent_features.png         # Bar plot of top 25 prominent features
        ├── lofo_top_features_accuracy_drop.png # Bar plot of single features causing accuracy drop
        └── cluster_ablation_accuracy_drop.png # Bar plot of collective clusters causing accuracy drop
```

---

## 🚀 Quick Start

### 1. Installation

Using Conda (recommended for OpenMM and GPU acceleration):

```bash
git clone https://github.com/N-S8990/MD-ContactML.git
cd MD-ContactML
conda env create -f environment.yml
conda activate md-contactml
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 2. Run Molecular Dynamics Simulation

Simulate all 3 independent replicas across the temperature series (300 K vs. 400 K) with OpenMM:

```bash
# Simulates replica_1, replica_2, and replica_3 into data/ (200 frames per trajectory)
python md_simulation.py \
  --input_pdb 1A22.pdb \
  --out_dir data \
  --num_replicas 3 \
  --temperatures 300 400 \
  --steps 200000 \
  --platform CUDA
```

*(Tip: You can simulate a specific replica on a separate GPU by passing `--replica_id 1`, or pass `--skip_existing` to avoid re-simulating completed runs).*

### 3. Run the Feature Extraction & ML Pipeline

Run the full end-to-end pipeline across 3 independent replicas per condition:

```bash
python run_pipeline.py \
  --class0_top data/replica_1/temp_300K/sim_300K_prepared.pdb \
  --class0_traj data/replica_1/temp_300K/sim_300K_traj.dcd data/replica_2/temp_300K/sim_300K_traj.dcd data/replica_3/temp_300K/sim_300K_traj.dcd \
  --class1_top data/replica_1/temp_400K/sim_400K_prepared.pdb \
  --class1_traj data/replica_1/temp_400K/sim_400K_traj.dcd data/replica_2/temp_400K/sim_400K_traj.dcd data/replica_3/temp_400K/sim_400K_traj.dcd \
  --corr_threshold 0.98 \
  --acc_tolerance 0.05 \
  --out_dir results
```

*(Tip: Add `--skip_lofo` or `--skip_cluster_lofo` for fast runs during development).*

#### CLI Arguments Reference:
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--class0_top` | required | Condition-0 topology (`.pdb`) |
| `--class0_traj` | required | Independent condition-0 replica trajectories (minimum 3) |
| `--class1_top` | required | Condition-1 topology (`.pdb`) |
| `--class1_traj` | required | Independent condition-1 replica trajectories (minimum 3) |
| `--corr_threshold` | `0.98` | Pearson correlation coefficient threshold for collinearity elimination |
| `--acc_tolerance` | `0.05` | Maximum allowable validation accuracy drop before stopping pruning |
| `--run_cluster_lofo`| `True` | Execute Group-Level Cluster Ablation across residue pairs and systems |
| `--skip_cluster_lofo`| flag | Skip the Group Cluster Ablation study |
| `--run_lofo` | `True` | Execute Leave-One-Feature-Out (LOFO) ablation testing |
| `--skip_lofo` | flag | Skip the LOFO ablation study to save runtime |
| `--lofo_min_drop` | `0.002` | Minimum accuracy drop threshold to list ablated features/clusters |
| `--max_frames` | `None` | Max frames to read per trajectory (useful for quick testing) |
| `--frame_stride` | `1` | Keep every Nth frame; reduces time-series autocorrelation |
| `--out_dir` | `results` | Base directory for result tables, figures, and reports |

---

## 📊 Benchmark Performance (Held-Out Test Replica 3)

| Classifier | Test Accuracy | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **98.00%** | **98.00%** | **1.0000** | **0.9600** | **0.9796** | **0.9992** |
| **Neural Net (MLP)** | **96.50%** | 96.50% | 0.9745 | 0.9550 | 0.9646 | 0.9958 |
| **Logistic Regression** | **95.75%** | 95.75% | 0.9598 | 0.9550 | 0.9574 | 0.9936 |

### 🔬 Key Mechanistic Discoveries & Redundancy Unmasking

By coupling single-feature LOFO with **Group-Level Cluster Ablation** (where all representations of a contact pair or subsystem are removed simultaneously), the framework successfully overcomes **redundancy masking** (where sister features like $C_\beta$ or soft contact hide the loss of $d_{\min}$):

1. **Primary Interface Latch (`Glu163 - Arg172`)**:
   - Single-feature LOFO drop: **-0.25%**
   - **Collective Cluster Ablation Drop: -2.50%** ($97.50\% \to 95.00\%$, log-loss surge $+0.0159$).
   - Rank #1 master impact hotspot; acts as the primary electrostatic/H-bond latch stabilizing the complex.
2. **Central Hydrophobic & Steric Clamp (`Thr164 - Gly123 / Trp124`)**:
   - Both pairs exhibit a sharp **-1.25% collective accuracy drop** when ablated as whole biological units.
3. **Unmasking Macro-Conformational Expansion (`CLUSTER_GLOBAL_SHAPE`)**:
   - In single-feature LOFO, `GLOBAL_Rg_A` dropped 0.00% because $R_{g,\text{complex}}$ served as an immediate backup.
   - In Group LOFO, ablating all 5 global shape features produces a **-0.75% accuracy drop** and the highest test log-loss surge in the system (**+0.0178**).
4. **Canonical Receptor Anchor (`Arg64 - Trp124`)**:
   - Previously showed 0.00% drop under single LOFO; ablating its 9 multi-modal features together uncovers a **-0.75% accuracy drop**.
5. **Cation–$\pi$ & Dynamic Networks**:
   - `Arg172(B) - Phe165(A)` cation–$\pi$ cluster causes a **-1.00% accuracy drop**.
   - `CLUSTER_GRAPH_TOPOLOGY` and `CLUSTER_INTERFACE_PACKING` each cause **-0.50% accuracy drops**, proving dynamic network rewiring is an autonomous causal signal of thermal perturbation.

---

*Built for computational biophysics and structural bioinformatics research.*
