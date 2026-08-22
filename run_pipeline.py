import argparse
import logging
import os
import shutil
import pandas as pd

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
    parser.add_argument("--cov_traj", type=str, default="data/current_sim/sim_traj.dcd", help="Trajectory file")
    parser.add_argument("--cov_top", type=str, default="data/current_sim/sim_prepared.pdb", help="Topology file")
    parser.add_argument("--cov2_traj", type=str, default="data/current_sim/sim_traj.dcd", help="Trajectory file 2")
    parser.add_argument("--cov2_top", type=str, default="data/current_sim/sim_prepared.pdb", help="Topology file 2")
    
    parser.add_argument("--corr_threshold", type=float, default=0.90, help="Correlation threshold to drop features")
    parser.add_argument("--acc_tolerance", type=float, default=0.05, help="Max accuracy drop allowed before stopping")
    parser.add_argument("--min_features", type=int, default=10, help="Minimum number of features to keep")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to read per trajectory (for testing)")
    
    parser.add_argument("--out_dir", type=str, default="results", help="Base output directory")
    
    args = parser.parse_args()
    
    if os.path.exists(args.out_dir):
        logger.info(f"Removing older results in {args.out_dir}...")
        shutil.rmtree(args.out_dir)
        
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    
    logger.info(f"Saving results to {args.out_dir}")
    
    logger.info("=== Phase 1: Feature Extraction ===")
    # The Zenodo dataset lacks segids, so we will just compute distances between the first 200 CA atoms 
    # and the next 600 CA atoms as a proxy for RBD-ACE2 interactions to test the pipeline.
    extractor = MDFeatureExtractor(
        target1_selection="segid A and name CA",
        target2_selection="segid B and name CA",
        target1_name="HGH",
        target2_name="RECEPTOR"
    )
    cov_traj_all = [args.cov_traj]
    cov2_traj_all = [args.cov2_traj]
    
    X, y = extractor.build_dataset(args.cov_top, cov_traj_all, args.cov2_top, cov2_traj_all, max_frames=args.max_frames)
    
    # Keep the top 1000 features representing the closest atom pairs (the actual interface)
    logger.info("Selecting the 1000 closest interacting atom pairs to use as features...")
    mean_distances = X.mean()
    closest_1000_cols = mean_distances.nsmallest(1000).index
    X = X[closest_1000_cols]
    
    # Save original features for reference
    X_orig = X.copy()
    
    logger.info("=== Phase 2: Preprocessing ===")
    preprocessor = MDPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(X, y)
    
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
    
    with open(os.path.join(args.out_dir, "final_features.txt"), "w") as f:
        f.write("\n".join(final_features))
        
    logger.info("=== Phase 5: Visualization ===")
    vis = PipelineVisualizer(output_dir=os.path.join(args.out_dir, "figures"))
    
    # Plot accuracy curve
    vis.plot_accuracy_vs_features(log_df)
    
    # Plot final features
    vis.plot_surviving_features(final_features)
    
    # Heatmaps (before vs after)
    vis.plot_correlation_heatmap(X_orig[final_features], "Final Feature Correlation Matrix", "heatmap_after.png")
    
    logger.info("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
