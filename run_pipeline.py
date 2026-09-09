import argparse
import logging
import os
import shutil
import pandas as pd
from pathlib import Path

from ml_pipeline import (
    MDFeatureExtractor,
    MDPreprocessor,
    FeatureEliminationLoop,
    PipelineVisualizer
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
    extractor = MDFeatureExtractor(
        target1_selection=args.target1_selection,
        target2_selection=args.target2_selection,
        target1_name=args.target1_name,
        target2_name=args.target2_name
    )
    X, y, groups = extractor.build_dataset(
        args.class0_top, args.class0_traj, args.class1_top, args.class1_traj,
        max_frames=args.max_frames, frame_stride=args.frame_stride
    )
    
    # Keep the top 5000 features representing the closest atom pairs (the actual interface)
    # logger.info("Selecting the 5000 closest interacting atom pairs to use as features...")
    # mean_distances = X.mean()
    # closest_5000_cols = mean_distances.nsmallest(1000).index
    # X = X[closest_5000_cols]
    
    # Save original features for reference
    X_orig = X.copy()
    
    logger.info("=== Phase 2: Preprocessing ===")
    preprocessor = MDPreprocessor(n_splits=args.cv_folds, test_fold=args.test_fold)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(X, y, groups)
    
    logger.info("=== Phase 3 & 4: Iterative Elimination Loop ===")
    loop = FeatureEliminationLoop(
        corr_threshold=args.corr_threshold,
        accuracy_tolerance=args.acc_tolerance,
        min_features=args.min_features
    )
    
    final_features, log_df = loop.run(X_train, X_test, y_train, y_test, model_to_track='rf')
    
    # Save logs
    log_path = os.path.join(args.out_dir, "elimination_log.csv")
    log_df.to_csv(log_path, index=False)
    logger.info(f"Saved elimination log to {log_path}")
    
    # Save final features as CSV
    final_features_df = pd.DataFrame({'feature': final_features})
    final_features_path = os.path.join(args.out_dir, "final_features.csv")
    final_features_df.to_csv(final_features_path, index=False)
    logger.info(f"Saved final features to {final_features_path}")
        
    # Extract final accuracy and recall
    final_acc = log_df.iloc[-1]['rf_accuracy']
    final_recall = log_df.iloc[-1]['rf_recall']
    
    score_text = f"Final Model Scores (Random Forest):\nAccuracy: {final_acc:.4f}\nRecall: {final_recall:.4f}\n"
    logger.info(f"\n{'-'*40}\n{score_text}{'-'*40}")
    
    final_scores_path = os.path.join(args.out_dir, "final_scores.txt")
    with open(final_scores_path, "w") as f:
        f.write(score_text)
    logger.info(f"Saved final scores to {final_scores_path}")
        
    logger.info("=== Phase 5: Visualization ===")
    vis = PipelineVisualizer(output_dir=os.path.join(args.out_dir, "figures"))
    
    # Plot accuracy curve
    vis.plot_accuracy_vs_features(log_df)
    

    
    # Heatmaps (before vs after)
    vis.plot_correlation_heatmap(X_orig[final_features], "Final Feature Correlation Matrix", "heatmap_after.png")
    
    logger.info("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
