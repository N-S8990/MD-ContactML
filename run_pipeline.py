import argparse
import logging
import os
import shutil
import time
import pandas as pd
from pathlib import Path

from ml_pipeline import (
    MDFeatureExtractor,
    MDPreprocessor,
    FeatureEliminationLoop,
    PipelineVisualizer,
    HAS_CUML
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="MD-ContactML: Iterative Feature Elimination for MD Trajectories")
    parser.add_argument("--class0_traj", nargs="+", required=True,
                        help="Independent condition-0 replica trajectories (minimum three).")
    parser.add_argument("--class0_top", required=True, help="Topology for condition 0.")
    parser.add_argument("--class1_traj", nargs="+", required=True,
                        help="Independent condition-1 replica trajectories (minimum three).")
    parser.add_argument("--class1_top", required=True, help="Topology for condition 1.")
    parser.add_argument("--target1_selection", default="chainID A and name CA",
                        help="MDAnalysis selection for complex partner 1.")
    parser.add_argument("--target2_selection", default="chainID B and name CA",
                        help="MDAnalysis selection for complex partner 2.")
    parser.add_argument("--target1_name", default="PARTNER1", help="Label used in contact feature names.")
    parser.add_argument("--target2_name", default="PARTNER2", help="Label used in contact feature names.")
    parser.add_argument("--corr_threshold", type=float, default=0.90, help="Correlation threshold to drop features")
    parser.add_argument("--acc_tolerance", type=float, default=0.05, help="Max accuracy drop allowed before stopping")
    parser.add_argument("--min_features", type=int, default=10, help="Minimum number of features to keep")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to read per trajectory (for testing)")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Keep every Nth trajectory frame (default: 1).")
    parser.add_argument("--cv_folds", type=int, default=3,
                        help="Number of replica-aware folds; requires this many replicas per class.")
    parser.add_argument("--test_fold", type=int, default=0,
                        help="Which replica-aware fold to reserve for testing (0-based).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for trajectory feature extraction.")
    parser.add_argument("--initial-features", type=int, default=None,
                        help="Optional: Pre-filter to N closest features based on mean distance before correlation elimination.")
    
    parser.add_argument("--out_dir", type=str, default="results", help="Base output directory")
    
    args = parser.parse_args()

    class0_paths = {Path(path).resolve() for path in args.class0_traj}
    class1_paths = {Path(path).resolve() for path in args.class1_traj}
    shared_paths = class0_paths & class1_paths
    if shared_paths:
        parser.error("A trajectory cannot appear in both classes: " + ", ".join(map(str, shared_paths)))
    if args.cv_folds > 1 and (len(args.class0_traj) < args.cv_folds or len(args.class1_traj) < args.cv_folds):
        parser.error(f"For cv_folds={args.cv_folds}, provide at least {args.cv_folds} independent trajectories for each class. Or use --cv_folds 1 for a single trajectory per class.")
    
    if os.path.exists(args.out_dir):
        logger.info(f"Removing older results in {args.out_dir}...")
        shutil.rmtree(args.out_dir)
        
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    
    logger.info(f"Saving results to {args.out_dir}")
    
    logger.info("=== Phase 1: Feature Extraction ===")
    t0 = time.time()
    extractor = MDFeatureExtractor(
        target1_selection=args.target1_selection,
        target2_selection=args.target2_selection,
        target1_name=args.target1_name,
        target2_name=args.target2_name
    )
    X, y, groups = extractor.build_dataset(
        args.class0_top, args.class0_traj, args.class1_top, args.class1_traj,
        max_frames=args.max_frames, frame_stride=args.frame_stride, workers=args.workers
    )
    t_extract = time.time() - t0
    
    # Save original features for reference
    X_orig = X.copy()
    original_feature_count = X.shape[1]
    
    logger.info("=== Phase 2: Preprocessing ===")
    t0 = time.time()
    preprocessor = MDPreprocessor(n_splits=args.cv_folds, test_fold=args.test_fold, initial_features=getattr(args, 'initial_features', None))
    X_train, X_test, y_train, y_test, groups_train, groups_test = preprocessor.preprocess(X, y, groups)
    t_preprocess = time.time() - t0
    
    initial_feature_count = getattr(args, 'initial_features', None) if getattr(args, 'initial_features', None) is not None else original_feature_count
    
    logger.info("=== Phase 3: Model Setup & Hyperparameter Search ===")
    t0 = time.time()
    loop = FeatureEliminationLoop(
        corr_threshold=args.corr_threshold,
        accuracy_tolerance=args.acc_tolerance,
        min_features=args.min_features
    )
    # Tune hyperparameters on the training data BEFORE elimination
    loop.trainer.tune_hyperparameters(X_train, y_train, groups=groups_train, cv_folds=args.cv_folds)
    t_tune = time.time() - t0
    
    logger.info("=== Phase 4: Iterative Elimination Loop ===")
    t0 = time.time()
    final_features, log_df = loop.run(X_train, y_train, groups_train, X_test, y_test, model_to_track='rf', cv_folds=args.cv_folds)
    t_eliminate = time.time() - t0
    
    # Save logs
    log_path = os.path.join(args.out_dir, "elimination_log.csv")
    log_df.to_csv(log_path, index=False)
    logger.info(f"Saved elimination log to {log_path}")
    
    # Save final features as CSV
    final_features_df = pd.DataFrame({'feature': final_features})
    final_features_path = os.path.join(args.out_dir, "final_features.csv")
    final_features_df.to_csv(final_features_path, index=False)
    logger.info(f"Saved final features to {final_features_path}")
    
    logger.info("=== Phase 5: Visualization ===")
    vis = PipelineVisualizer(output_dir=os.path.join(args.out_dir, "figures"))
    vis.plot_accuracy_vs_features(log_df)
    vis.plot_correlation_heatmap(X_orig[final_features], "Final Feature Correlation Matrix", "heatmap_after.png")
    
    logger.info("=== BENCHMARK & PERFORMANCE REPORT ===")
    
    final_row = log_df.iloc[-1]
    
    report = [
        "TIMING:",
        f"Feature Extraction Time:  {t_extract:.2f}s",
        f"Preprocessing Time:       {t_preprocess:.2f}s",
        f"Hyperparameter Tuning:    {t_tune:.2f}s",
        f"Feature Elimination Time: {t_eliminate:.2f}s",
        f"Total Pipeline Time:      {t_extract + t_preprocess + t_tune + t_eliminate:.2f}s",
        "",
        "CONFIGURATION:",
        f"Workers:                  {args.workers}",
        f"cuML / GPU RF Available:  {HAS_CUML}",
        f"RF Params:                {loop.trainer.models['rf'].get_params() if not HAS_CUML else 'cuML defaults'}",
        f"Original Features:        {original_feature_count}",
        f"Initial Features Used:    {initial_feature_count}",
        f"Final Features Selected:  {len(final_features)}",
        "",
        "FINAL METRICS (TEST SET):"
    ]
    
    for model_key in ['rf', 'lr', 'mlp']:
        report.append(f"  {model_key.upper()}:")
        report.append(f"    Accuracy:          {final_row.get(f'{model_key}_accuracy', float('nan')):.4f}")
        report.append(f"    Balanced Accuracy: {final_row.get(f'{model_key}_balanced_accuracy', float('nan')):.4f}")
        report.append(f"    Precision:         {final_row.get(f'{model_key}_precision', float('nan')):.4f}")
        report.append(f"    Recall:            {final_row.get(f'{model_key}_recall', float('nan')):.4f}")
        report.append(f"    F1:                {final_row.get(f'{model_key}_f1_score', float('nan')):.4f}")
        report.append(f"    ROC-AUC:           {final_row.get(f'{model_key}_roc_auc', float('nan')):.4f}")
    
    score_text = "\n".join(report)
    logger.info(f"\n{'-'*50}\n{score_text}\n{'-'*50}")
    
    final_scores_path = os.path.join(args.out_dir, "benchmark_report.txt")
    with open(final_scores_path, "w") as f:
        f.write(score_text)
    logger.info(f"Saved benchmark report to {final_scores_path}")
    
    logger.info("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
