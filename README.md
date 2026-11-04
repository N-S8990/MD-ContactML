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
│ (OpenMM / PDBFixer)    │     │ (Multi-Modal Features) │     │   (Threshold = 0.98)   │
└────────────────────────┘     └────────────────────────┘     └────────────────────────┘
            │                                                              │
            ▼                                                              ▼
┌────────────────────────┐                                    ┌────────────────────────┐
│  Zero-Leakage Split    │                                    │  LOFO Empirical Ablat. │
│(Train: Rep1-4 |Test: 5)│                                    │  (Hotspot Discovery)   │
└────────────────────────┘                                    └────────────────────────┘
            │                                                              │
            ▼                                                              ▼
┌────────────────────────┐                                    ┌────────────────────────┐
│  Model Benchmarking    │ ─────────────────────────────────> │  Publication Artifacts │
│  (RF, MLP, Logistic)   │                                    │ (Curves, Heatmaps, CSV)│
└────────────────────────┘                                    └────────────────────────┘
```

1. **Structure Preparation & MD Simulation**: Automated cleaning, missing atom/hydrogen addition, energy minimization, gentle warming, and production MD simulations via OpenMM.
2. **Multi-Modal Biophysical Feature Extraction**:
   - **Buried Surface Area ($\Delta\text{SASA}$)** & Solvent Accessibility: Total buried surface area ($\text{SASA}_A + \text{SASA}_B - \text{SASA}_{AB}$), partitioned into hydrophobic/apolar and polar burial via fast Fibonacci sphere integration.
   - **Strict Directional Hydrogen Bonds**: Donor–Hydrogen $\cdots$ Acceptor angular filters ($\theta \ge 120^\circ$ and $\theta \ge 135^\circ$) decomposed into Backbone–Backbone, Sidechain–Sidechain, and Sidechain–Backbone pairs.
   - **Aligned Per-Residue Dynamics**: Rigid-body-aligned Cα RMSD, trajectory RMSF, instantaneous reference displacement, and RMSF-derived B-factor equivalents: $B = \frac{8\pi^2}{3}\mathrm{RMSF}^2$.
   - **Interface Geometry & Planarity**: Least-squares plane thickness ($\lambda_3 / \sum \lambda$), eccentricity, and span radius of the interfacial contact surface.
   - **Weighted Network Graph Topology**: Continuous contact edge weights ($1/d_{\min}$), weighted node strength, distance-weighted communication efficiency, and centrality metrics.
   - **Backbone ($C_\alpha - C_\alpha$) & Sidechain Orientation ($C_\beta - C_\beta$) Distances**.
   - **Minimum Heavy-Atom Contact Distances ($d_{\min}$)** & sidechain centroid (COM) packing vectors.
   - **Continuous Soft Contact Switching Functions**: $s(r) = (1 - (r/5)^6) / (1 - (r/5)^{12}) \in [0, 1]$.
   - **Inverse Proximities**: $1 / (d_{\min} + 0.01)$.
   - **Cation–$\pi$ Interactions**: Direct geometric tracking and continuous contact scoring between basic residues (Arg, Lys) and aromatic rings (Trp, Phe, Tyr, His).
   - **Directional Hydrogen Bonds**: Distance and angular geometric tracking between specific and global donor–acceptor pairs.
   - **Molecular Dipole Alignment**: Dipole vector norms ($\|\boldsymbol{\mu}_A\|$, $\|\boldsymbol{\mu}_B\|$), alignment angle ($\cos\theta$), and cross-partner charge centroid distances.
   - **Interface Packing Density**: Inter-chain vs. intra-chain interface contact counts and packing ratio.
   - **Salt Bridges & Aromatic $\pi$-Stacking**: Explicit ion pairs and aromatic ring centroid proximities.
   - **Dynamic Network Graph Topology**: Average/max degree, betweenness centrality, degree centrality, clustering coefficient, transitivity, and Laplacian algebraic connectivity ($\lambda_2$).
   - **Backbone & Sidechain Dihedrals**: $\sin$ and $\cos$ components for backbone $\phi, \psi, \omega$ and sidechain rotamer $\chi_1$ on both Partner A and Partner B.
   - **Global Shape & Dynamics**: Radii of gyration ($R_g$), shape asphericity, inter-chain COM distance, and backbone RMSD.
3. **Zero-Data-Leakage Replica Splitting**:
   - Replicas 1 and 2 are used for training and inner validation.
   - Replica 5 is strictly held-out as an independent test set; validation and feature discovery use whole training replicas.
   - Standard scaling is fitted strictly on training frames.
4. **Iterative Collinearity Elimination Loop**:
   - Computes the Pearson correlation matrix on training frames only ($|r|$).
   - Iteratively detects and prunes collinear pairs ($|r| \ge 0.98$) in batches.
   - Gated by inner validation accuracy tolerance ($5\%$), stopping if pruning degrades performance.
5. **Multi-Model Benchmark**: Evaluates Random Forest, Logistic Regression, and MLP on the held-out test replica.
6. **Multi-Scale Empirical Ablation (LOFO & Group Cluster Ablation)**:
   - **Grouped Features and LOFO**: Correlated mathematical representations are collapsed into biological residue-pair or subsystem groups before modeling and ablation.
   - **Group-Level Cluster Ablation**: Simultaneously ablates all mathematical representations belonging to a specific residue–residue contact pair (e.g. $C_\alpha$, $C_\beta$, $d_{\min}$, soft contact, inverse contact, H-bonds) or functional biophysical subsystem (e.g. `CLUSTER_GLOBAL_SHAPE`, `CLUSTER_GRAPH_TOPOLOGY`, `CLUSTER_INTERFACE_PACKING`). This eliminates "redundancy masking" and isolates the true collective causal necessity of each biophysical element.
   - **Grouped Permutation Importance**: Permutes correlated biological groups together on an inner validation replica.
   - **Distribution Effects**: Reports class-wise means and standardized mean differences independently of classifier importance.

---

## 📁 Project Structure

```
MD-ContactML/
├── md_simulation.py        # OpenMM MD simulation and structure preparation script
├── ml_pipeline.py          # Core library (feature extraction, ML models, elimination, LOFO, plotting)
├── run_pipeline.py         # Single unified CLI driver running the full workflow end-to-end
├── README.md               # Complete project documentation
├── requirements.txt        # Pip dependencies for analysis and simulation
├── environment.yml         # Conda environment definition
├── 1A22.pdb                # Reference Growth Hormone - Receptor complex structure
├── data/                   # Simulation trajectories (300 K & 400 K, Replicas 1-5)
└── results/                # Central output directory (reports, CSV data, figures)
    ├── pipeline_report.txt               # Plaintext executive summary report
    ├── raw_features.csv                  # Full feature matrix (size depends on input)
    ├── run_metadata.json                 # Command, seed, environment, and input metadata
    ├── train_correlation_matrix.csv      # Training-only Pearson correlation matrix
    ├── elimination_log.csv               # Collinearity elimination logs
    ├── model_benchmark_metrics.csv       # RF, Logistic Regression, MLP metrics
    ├── group_cv_metrics.csv              # Group-aware training cross-validation metrics
    ├── test_metric_confidence_intervals.csv # Bootstrap intervals on held-out metrics
    ├── prominent_features.csv            # Features ranked by RF + Permutation importance
    ├── grouped_permutation_importance.csv # Grouped importance on inner validation replica
    ├── feature_distribution_effects.csv  # Class-wise means and standardized effects
    ├── lofo_feature_importance.csv       # Single features ranked by empirical ablation accuracy drop
    ├── cluster_ablation_importance.csv   # Biophysical clusters ranked by collective ablation accuracy drop
    ├── correlation_matrices/
    │   ├── corr_matrix_initial.csv       # Initial correlation before pruning
    │   └── corr_matrix_final.csv         # Final correlation after pruning
    └── figures/
        ├── train_correlation_heatmap.png      # Training-only feature correlation heatmap
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

The pip requirements include OpenMM and install PDBFixer from its upstream repository. GPU acceleration still requires a compatible local OpenMM platform; use `--platform CPU` when CUDA or OpenCL is unavailable.

### 2. Run Molecular Dynamics Simulation

Simulate all 5 independent replicas across the temperature series (300 K vs. 400 K) with OpenMM:

```bash
# Simulates replica_1 through replica_5 into data/ (200 frames per trajectory)
python md_simulation.py \
  --input_pdb 1A22.pdb \
  --out_dir data \
  --num_replicas 5 \
  --temperatures 300 400 \
  --steps 200000 \
  --platform CUDA
```

*(Tip: You can simulate a specific replica on a separate GPU by passing `--replica_id 1`, or pass `--skip_existing` to avoid re-simulating completed runs).*

### 3. Run the Feature Extraction & ML Pipeline

Run the full end-to-end pipeline across 5 independent replicas per condition:

```bash
python run_pipeline.py \
  --class0_top data/replica_1/temp_300K/sim_300K_prepared.pdb \
  --class0_traj data/replica_1/temp_300K/sim_300K_traj.dcd data/replica_2/temp_300K/sim_300K_traj.dcd data/replica_3/temp_300K/sim_300K_traj.dcd data/replica_4/temp_300K/sim_300K_traj.dcd data/replica_5/temp_300K/sim_300K_traj.dcd \
  --class1_top data/replica_1/temp_400K/sim_400K_prepared.pdb \
  --class1_traj data/replica_1/temp_400K/sim_400K_traj.dcd data/replica_2/temp_400K/sim_400K_traj.dcd data/replica_3/temp_400K/sim_400K_traj.dcd data/replica_4/temp_400K/sim_400K_traj.dcd data/replica_5/temp_400K/sim_400K_traj.dcd \
  --corr_threshold 0.98 \
  --acc_tolerance 0.05 \
  --out_dir results
```

*(Tip: Add `--skip_lofo` or `--skip_cluster_lofo` for fast runs during development).*

For a quick smoke run, use `--max_frames 2 --skip_lofo --skip_cluster_lofo --permutation_repeats 0`. Feature extraction is cached under `results/cache` by default; use `--no_cache` to disable caching. Use `--workers N` to parallelize trajectory extraction.

#### CLI Arguments Reference:
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--class0_top` | required | Condition-0 topology (`.pdb`) |
| `--class0_traj` | required | Independent condition-0 replica trajectories (minimum 5) |
| `--class1_top` | required | Condition-1 topology (`.pdb`) |
| `--class1_traj` | required | Independent condition-1 replica trajectories (minimum 5) |
| `--test_replica` | `5` | Whole replica group reserved for final testing |
| `--corr_threshold` | `0.98` | Pearson correlation coefficient threshold for collinearity elimination |
| `--acc_tolerance` | `0.05` | Maximum allowable validation accuracy drop before stopping pruning |
| `--skip_cluster_lofo`| flag | Skip the Group Cluster Ablation study |
| `--skip_lofo` | flag | Skip the LOFO ablation study to save runtime |
| `--lofo_min_drop` | `0.01` | Minimum accuracy drop threshold to list ablated features/clusters (>1% by default) |
| `--max_frames` | `None` | Max frames to read per trajectory (useful for quick testing) |
| `--frame_stride` | `1` | Keep every Nth frame; reduces time-series autocorrelation |
| `--workers` | `1` | Parallel trajectory-extraction workers |
| `--no_cache` | flag | Disable cached trajectory feature extraction |
| `--cv_folds` | `4` | Number of group-aware folds over training replicas |
| `--bootstrap_samples` | `200` | Bootstrap samples for held-out metric confidence intervals |
| `--permutation_repeats` | `10` | Permutation-importance repeats; `0` disables it |
| `--seed` | `42` | Reproducibility seed |
| `--out_dir` | `results` | Base directory for result tables, figures, and reports |

Run `python -m py_compile ml_pipeline.py run_pipeline.py md_simulation.py` as a quick syntax check before a long run.

---

## 📊 Generated Results

Performance values are generated from the current five-replica run and should not be copied from an older run. The final benchmark is the held-out replica group selected by `--test_replica` (default: replica 5).

Results are generated into CSV files by each run. The benchmark includes accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log-loss, and Brier score.

---

*Built for computational biophysics and structural bioinformatics research.*
