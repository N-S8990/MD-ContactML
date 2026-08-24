# 🧬 MD-ContactML

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenMM](https://img.shields.io/badge/OpenMM-MD%20Simulation-3776AB?logo=python&logoColor=white)
![PyMOL](https://img.shields.io/badge/PyMOL-3D%20Rendering-009688)
![MDAnalysis](https://img.shields.io/badge/MDAnalysis-Trajectory%20Processing-brightgreen)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)

**MD-ContactML** is an end-to-end computational biophysics and machine learning framework for simulating macromolecular complexes, extracting dynamic residue–residue contact networks at protein–protein interfaces, rendering 3D complex structures, and performing iterative feature elimination to isolate the most critical, non-redundant biophysical interactions.

---

## 🛠 Tech Stack

- **Molecular Dynamics & Structure Prep:** [OpenMM](https://openmm.org/), [PDBFixer](https://github.com/openmm/pdbfixer) (Amber14 forcefield, PME, Langevin dynamics, explicit solvation)
- **Data Processing & MD Trajectories:** [MDAnalysis](https://www.mdanalysis.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [SciPy](https://scipy.org/)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/) (Random Forest, Multi-Layer Perceptron Neural Networks, Logistic Regression)
- **3D Structure Visualization:** [PyMOL](https://pymol.org/) (Headless automated cartoon rendering and session export)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Utilities:** [tqdm](https://tqdm.github.io/) (Progress Tracking), [PyYAML](https://pyyaml.org/)

---

## 🔬 Pipeline Overview

In molecular dynamics simulations of viral complexes (such as the SARS-CoV-2 Spike RBD complexed with the human ACE2 receptor), thousands of pairwise atomic distances fluctuate dynamically across frames. Many interface contact distances exhibit high collinearity and redundancy.

**MD-ContactML** provides a complete modular workflow:

```
┌────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│  Structure Prep & MD   │ ──> │   Feature Extraction   │ ──> │  Iterative Elimination │
│ (OpenMM / PDBFixer)    │     │  (MDAnalysis Distances)│     │   (Correlation + ML)   │
└────────────────────────┘     └────────────────────────┘     └────────────────────────┘
            │                                                              │
            ▼                                                              ▼
┌────────────────────────┐                                    ┌────────────────────────┐
│  3D PyMOL Rendering    │                                    │  Plots & Results Logs  │
│  (.png render & .pse)  │                                    │ (Curves, Heatmaps, CSV)│
└────────────────────────┘                                    └────────────────────────┘
```

1. **Structure Preparation & MD Simulation**: Automated cleaning, missing atom/hydrogen addition, solvation, energy minimization, and production MD simulation via OpenMM.
2. **3D Structural Rendering**: Automated headless PyMOL pipeline generating publication-ready cartoon figures and `.pse` session files.
3. **Contact Feature Extraction**: Pairwise $C_\alpha$ interface distance extraction across frames and replicas. Top closest interacting pairs are selected as interface candidate features.
4. **Iterative Elimination Loop**:
   - Detects highly correlated feature pairs ($r \ge 0.90$).
   - Identifies and eliminates globally redundant contact features.
   - Concurrently monitors ML classifier performance (Random Forest, MLP, Logistic Regression) with accuracy tolerance gates.
5. **Visualization & Reporting**: Outputs accuracy decay curves, correlation heatmaps before/after pruning, surviving contact networks, and step-by-step elimination logs.

---

## 📁 Project Structure

```text
MD-ContactML/
├── 6m0j.pdb                 # Reference SARS-CoV-2 RBD / ACE2 complex structure
├── md_simulation.py         # OpenMM MD simulation and structure preparation script
├── ml_pipeline.py           # Core library (feature extraction, ML models, elimination logic, plotting)
├── run_pipeline.py          # Main execution driver for the ML contact elimination pipeline
├── requirements.txt         # Python package dependencies
├── renders/
│   └── render_pdb.py        # Automated PyMOL script for high-res rendering & session generation
├── data/
│   ├── current_sim/         # Generated simulation outputs (trajectories .dcd, prepared .pdb, logs)
│   └── trajectories/        # Trajectory storage
└── results/
    ├── result_1/            # Versioned run directories (auto-incremented)
    │   ├── elimination_log.csv  # Step-by-step feature removal and metric log
    │   ├── final_features.txt   # Surviving minimal non-redundant contact pairs
    │   └── figures/
    │       ├── accuracy_vs_features.png  # ML performance vs feature count
    │       ├── final_features.png        # Surviving feature frequency/distribution
    │       └── heatmap_after.png         # Correlation heatmap of final features
    └── result_2/            # Subsequent run outputs...
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/N-S8990/MD-ContactML.git
cd MD-ContactML
pip install -r requirements.txt
```

> [!TIP]
> For OpenMM with GPU acceleration (CUDA/OpenCL) or PyMOL rendering, ensure appropriate drivers and Conda/pip wheels are configured.

### 2. (Optional) Run Molecular Dynamics Simulation

Prepare the structure and run an all-atom MD simulation with OpenMM:

```bash
python md_simulation.py \
  --input_pdb 6m0j.pdb \
  --out_dir data/current_sim \
  --prefix sim \
  --steps 5000 \
  --platform CPU
```

Options:
- `--input_pdb`: Input PDB file path (default: `6m0j.pdb`).
- `--out_dir`: Destination directory for prepared PDB and DCD trajectory (default: `data/current_sim`).
- `--steps`: Number of MD integration steps (default: `5000`).
- `--platform`: Compute platform (`CPU`, `CUDA`, `OpenCL`, `Reference`).

### 3. (Optional) Render Structure with PyMOL

Generate a high-resolution 3D cartoon render and PyMOL session:

```bash
python renders/render_pdb.py 6m0j.pdb --output_name 6m0j_render
```

Outputs will be saved in `renders/`:
- `renders/6m0j_render.png`: 1200x900 150 DPI render.
- `renders/6m0j_render.pse`: PyMOL session file for interactive exploration.

### 4. Run the Feature Elimination & ML Pipeline

Run the contact extraction and iterative elimination loop:

```bash
python run_pipeline.py
```

### 5. Custom Pipeline Parameters

Configure trajectory inputs, thresholds, and convergence criteria:

```bash
python run_pipeline.py \
  --cov_traj data/current_sim/sim_traj.dcd \
  --cov_top data/current_sim/sim_prepared.pdb \
  --cov2_traj data/current_sim/sim_traj.dcd \
  --cov2_top data/current_sim/sim_prepared.pdb \
  --corr_threshold 0.85 \
  --acc_tolerance 0.05 \
  --min_features 10 \
  --out_dir results
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--cov_traj` | `data/current_sim/sim_traj.dcd` | Primary trajectory file (`.dcd` / `.pdb` / `.xtc`) |
| `--cov_top` | `data/current_sim/sim_prepared.pdb` | Primary topology file (`.pdb` / `.parm7`) |
| `--cov2_traj` | `data/current_sim/sim_traj.dcd` | Comparison / condition 2 trajectory file |
| `--cov2_top` | `data/current_sim/sim_prepared.pdb` | Comparison / condition 2 topology file |
| `--corr_threshold` | `0.90` | Pearson correlation coefficient threshold for redundancy |
| `--acc_tolerance` | `0.05` | Maximum allowable accuracy drop from baseline |
| `--min_features` | `10` | Minimum number of features to retain |
| `--max_frames` | `None` | Max frames to read per trajectory (useful for quick testing) |
| `--out_dir` | `results` | Base directory for auto-incrementing result folders (`result_1`, `result_2`, ...) |

---

## 🤖 Machine Learning Models

The pipeline benchmark evaluates dynamic contact networks across multiple model architectures:
- **Random Forest (RF)**: 100 decision trees with ensemble bagging (tracked model for elimination tolerance).
- **Multi-Layer Perceptron (MLP)**: Deep neural network with (128, 64) hidden layers, ReLU activation, and Adam optimization.
- **Logistic Regression (LR)**: Linear baseline with standard scaling and L-BFGS solver.

---

## 📊 Outputs & Artifacts

Each execution creates an isolated, versioned directory under `results/result_<N>/`:

- **`elimination_log.csv`**: Record of each elimination step, features removed, correlation values, and model scores.
- **`final_features.txt`**: List of surviving non-redundant residue contact pairs.
- **`figures/accuracy_vs_features.png`**: Accuracy trajectory across feature reduction steps.
- **`figures/final_features.png`**: Breakdown of surviving contact pairs.
- **`figures/heatmap_after.png`**: Correlation matrix heatmap of the final feature set.

---

*Built for computational biophysics and structural bioinformatics research.*
