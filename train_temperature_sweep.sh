#!/bin/bash

# Grouped Binary Classification
# Class 0 (Stable): 300K
# Class 1 (Unfolding): 400K

# Note: cv_folds is set to 1 because we have exactly 1 trajectory per class

python run_pipeline.py \
  --class0_top data/temp_300K/sim_300K_prepared.pdb \
  --class0_traj data/temp_300K/sim_300K_traj.dcd \
  --class1_top data/temp_400K/sim_400K_prepared.pdb \
  --class1_traj data/temp_400K/sim_400K_traj.dcd \
  --target1_selection "chainID A and name CA" \
  --target2_selection "chainID B and name CA" \
  --cv_folds 1 \
  --frame_stride 10 \
  --acc_tolerance 0.02 \
  --out_dir results_temperature_sweep
