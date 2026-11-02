#!/usr/bin/env python3

import argparse
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ml_pipeline import (
    ComprehensiveFeatureExtractor,
    MDPreprocessor,
    FeatureEliminationLoop,
    ModelTrainer,
    LOFOAblationEngine,
    GroupLOFOEngine,
    PipelineVisualizer,
    categorize_feature
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MD-ContactML")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MD-ContactML: Unified Biophysical Machine Learning Pipeline"
    )
    parser.add_argument("--class0_top", required=True, help="Topology (PDB) for Condition 0.")
    parser.add_argument("--class0_traj", nargs="+", required=True,
                        help="Independent replica trajectories for Condition 0 (e.g. 300 K).")
    parser.add_argument("--class1_top", required=True, help="Topology (PDB) for Condition 1.")
    parser.add_argument("--class1_traj", nargs="+", required=True,
                        help="Independent replica trajectories for Condition 1 (e.g. 400 K).")
    parser.add_argument("--corr_threshold", type=float, default=0.98,
                        help="Pearson correlation threshold to eliminate redundant features (default: 0.98).")
    parser.add_argument("--acc_tolerance", type=float, default=0.05,
                        help="Maximum accuracy drop allowed during elimination before stopping (default: 0.05).")
    parser.add_argument("--run_lofo", action="store_true", default=True,
                        help="Execute Leave-One-Feature-Out (LOFO) ablation study (default: True).")
    parser.add_argument("--skip_lofo", action="store_false", dest="run_lofo",
                        help="Skip the LOFO ablation study to save runtime.")
    parser.add_argument("--run_cluster_lofo", action="store_true", default=True,
                        help="Execute Group-Level LOFO (Cluster Ablation) study (default: True).")
    parser.add_argument("--skip_cluster_lofo", action="store_false", dest="run_cluster_lofo",
                        help="Skip the Group LOFO cluster ablation study.")
    parser.add_argument("--lofo_min_drop", type=float, default=0.002,
                        help="Minimum accuracy drop threshold to list ablated features in report/plot (default: 0.002 = 0.2%).")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max frames to extract per trajectory (optional, for testing).")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Stride between trajectory frames (default: 1).")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Directory to save all results, figures, and reports (default: results).")

    return parser.parse_args()


def main():
    args = parse_arguments()
    start_time = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visualizer = PipelineVisualizer(out_dir=out_dir)

    logger.info("=" * 80)
    logger.info("STARTING UNIFIED MD-CONTACTML PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Class 0 Replicas: {len(args.class0_traj)} | Class 1 Replicas: {len(args.class1_traj)}")
    logger.info(f"Correlation Threshold: {args.corr_threshold} | Output Directory: {out_dir}")

    t0_ext = time.time()
    extractor = ComprehensiveFeatureExtractor(reference_pdb=args.class0_top)
    X_full, y_full, groups_full = extractor.build_dataset(
        class0_top=args.class0_top,
        class0_trajs=args.class0_traj,
        class1_top=args.class1_top,
        class1_trajs=args.class1_traj,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride
    )
    extraction_time = time.time() - t0_ext
    logger.info(f"Feature Extraction Completed in {extraction_time:.2f}s.")
    X_full.to_csv(out_dir / "raw_features.csv", index=False)

    logger.info("Computing full Pearson Correlation Matrix across all features...")
    full_corr = X_full.corr().abs()
    full_corr.to_csv(out_dir / "full_correlation_matrix.csv")
    visualizer.plot_correlation_heatmap(
        full_corr,
        f"Full Correlation Matrix ({X_full.shape[1]} Features)",
        "full_correlation_heatmap.png"
    )

    upper_tri = full_corr.where(np.triu(np.ones(full_corr.shape), k=1).astype(bool))
    max_corr_series = np.maximum(
        upper_tri.max(axis=0).fillna(0),
        upper_tri.max(axis=1).fillna(0)
    )

    preprocessor = MDPreprocessor()
    test_group = f"rep_{len(args.class0_traj)}"
    X_train_scaled, y_train, groups_train, X_test_scaled, y_test = preprocessor.fit_transform_split(
        X_full, y_full, groups_full, test_group=test_group
    )
    logger.info(f"Train Set: {X_train_scaled.shape[0]} frames | Test Set ({test_group}): {X_test_scaled.shape[0]} frames")

    t0_elim = time.time()
    eliminator = FeatureEliminationLoop(
        corr_threshold=args.corr_threshold,
        accuracy_tolerance=args.acc_tolerance,
        out_dir=out_dir
    )
    active_features, log_df, corr_init, corr_final = eliminator.run(
        X_train_scaled, y_train, groups_train, X_test_scaled, y_test
    )
    elimination_time = time.time() - t0_elim

    visualizer.plot_correlation_heatmap(corr_init, "Initial Feature Correlation Matrix", "corr_matrix_initial.png")
    visualizer.plot_correlation_heatmap(corr_final, "Final Feature Correlation Matrix (Collinearity Removed)", "corr_matrix_final.png")
    visualizer.plot_elimination_trajectory(log_df, "elimination_trajectory.png")

    logger.info("Benchmarking classifiers on held-out test replica...")
    trainer = ModelTrainer()
    X_tr_final = X_train_scaled[active_features]
    X_te_final = X_test_scaled[active_features]

    benchmark_df = trainer.train_and_evaluate(X_tr_final, y_train, X_te_final, y_test)
    benchmark_df.to_csv(out_dir / "model_benchmark_metrics.csv")

    rf = trainer.models["Random Forest"]
    lr = trainer.models["Logistic Regression"]
    rf_importances = rf.feature_importances_
    lr_coefs = lr.coef_[0]

    logger.info("Computing Permutation Feature Importance on held-out test replica...")
    perm = permutation_importance(rf, X_te_final, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_importances = perm.importances_mean

    prominent_records = []
    for i, feat in enumerate(active_features):
        cat = categorize_feature(feat)
        prominent_records.append({
            "Feature": feat,
            "Category": cat,
            "RF_Importance": rf_importances[i],
            "Permutation_Importance": perm_importances[i],
            "LR_Coefficient": lr_coefs[i],
            "Shift_Direction_at_Class1": "Elevated / Expanded" if lr_coefs[i] > 0 else "Weakened / Contracted"
        })

    prom_df = pd.DataFrame(prominent_records)
    prom_df["Prominence_Score"] = 0.5 * (prom_df["RF_Importance"] / prom_df["RF_Importance"].max()) + \
                                  0.5 * (np.clip(prom_df["Permutation_Importance"], 0, None) / (prom_df["Permutation_Importance"].max() + 1e-6))
    prom_df = prom_df.sort_values(by="Prominence_Score", ascending=False).reset_index(drop=True)
    prom_df["Rank"] = prom_df.index + 1
    prom_df.to_csv(out_dir / "prominent_features.csv", index=False)
    visualizer.plot_prominent_features(prom_df, "top_prominent_features.png")

    cluster_df = None
    cluster_stats = None
    if args.run_cluster_lofo:
        t0_cl = time.time()
        logger.info("Running Group-Level LOFO (Cluster Ablation) across all functional & residue clusters...")
        group_engine = GroupLOFOEngine(out_dir=out_dir)
        cluster_df, cluster_stats = group_engine.run(
            X_train_scaled, y_train, X_test_scaled, y_test, prom_df=prom_df
        )
        visualizer.plot_cluster_ablation(cluster_df, "cluster_ablation_accuracy_drop.png", min_drop=args.lofo_min_drop)
        logger.info(f"Group LOFO Completed in {time.time() - t0_cl:.2f}s.")

    lofo_df = None
    lofo_stats = None
    if args.run_lofo:
        t0_lofo = time.time()
        logger.info("Running Leave-One-Feature-Out (LOFO) ablation testing across all features...")
        lofo_engine = LOFOAblationEngine(out_dir=out_dir)
        lofo_df, lofo_stats = lofo_engine.run(
            X_train_scaled, y_train, X_test_scaled, y_test, max_corr_series=max_corr_series
        )
        visualizer.plot_lofo_accuracy_drop(lofo_df, "lofo_top_features_accuracy_drop.png", min_drop=args.lofo_min_drop)
        logger.info(f"LOFO Ablation Completed in {time.time() - t0_lofo:.2f}s.")

    total_time = time.time() - start_time

    report_path = out_dir / "pipeline_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 85 + "\n")
        f.write("MD-CONTACTML: UNIFIED BIOPHYSICAL FEATURE & ML REPORT\n")
        f.write("=" * 85 + "\n\n")
        f.write(f"Total Execution Time:             {total_time:.2f} seconds\n")
        f.write(f"Feature Extraction Time:          {extraction_time:.2f} seconds\n")
        f.write(f"Elimination Loop Time:            {elimination_time:.2f} seconds\n")
        f.write(f"Correlation Threshold:            {args.corr_threshold}\n")
        f.write(f"Accuracy Tolerance:               {args.acc_tolerance}\n")
        f.write(f"Total Trajectory Frames:          {len(X_full)}\n")
        f.write(f"Total Initial Features:           {X_full.shape[1]}\n")
        f.write(f"Surviving Non-Redundant Features: {len(active_features)}\n")
        f.write(f"Elimination Steps:                {len(log_df) - 1}\n\n")

        f.write("-" * 85 + "\n")
        f.write("MODEL BENCHMARK PERFORMANCE (ON HELD-OUT TEST REPLICA)\n")
        f.write("-" * 85 + "\n")
        f.write(benchmark_df.to_string() + "\n\n")

        f.write("-" * 85 + "\n")
        f.write("TOP 20 MOST PROMINENT SURVIVING FEATURES (PRUNED MODEL)\n")
        f.write("-" * 85 + "\n")
        top20_prom = prom_df.head(20)[["Rank", "Feature", "Category", "Prominence_Score", "RF_Importance", "Permutation_Importance", "LR_Coefficient", "Shift_Direction_at_Class1"]]
        f.write(top20_prom.to_string(index=False) + "\n\n")

        if cluster_df is not None:
            drop_clusters = cluster_df[cluster_df["Accuracy_Drop"] >= args.lofo_min_drop]
            f.write("-" * 85 + "\n")
            f.write(f"ALL {len(drop_clusters)} FUNCTIONAL & RESIDUE CLUSTERS WITH ABLATION DROP >= {args.lofo_min_drop*100:.2f}%\n")
            f.write("-" * 85 + "\n")
            f.write(f"Cluster Baseline Accuracy: {cluster_stats['baseline_accuracy'] * 100:.2f}%\n")
            f.write("Rank | Cluster Name                   | Category                     | Acc Drop | Acc Without | Master Impact | Feats\n")
            f.write("-" * 85 + "\n")
            for idx, row in drop_clusters.iterrows():
                f.write(f"{row['Rank']:4d} | {row['Cluster'][:30]:30s} | {row['Category'][:28]:28s} | {row['Accuracy_Drop']*100:+7.2f}% | {row['Accuracy_Without_Cluster']*100:10.2f}% | {row['Master_Impact_Score']:13.4f} | {row['Num_Features']:5d}\n")
            f.write("-" * 85 + "\n\n")

        if lofo_df is not None:
            drop_feats = lofo_df[lofo_df["Accuracy_Drop"] >= args.lofo_min_drop]
            f.write("-" * 85 + "\n")
            f.write(f"ALL {len(drop_feats)} SINGLE FEATURES WITH LOFO ACCURACY DROP >= {args.lofo_min_drop*100:.2f}%\n")
            f.write("-" * 85 + "\n")
            f.write(f"LOFO Baseline Accuracy: {lofo_stats['baseline_accuracy'] * 100:.2f}%\n")
            f.write("Rank | Feature Name                   | Category                     | Acc Drop | Acc Without | Loss Increase | Max Corr\n")
            f.write("-" * 85 + "\n")
            for idx, row in drop_feats.iterrows():
                f.write(f"{row['Rank']:4d} | {row['Feature'][:30]:30s} | {row['Category'][:28]:28s} | {row['Accuracy_Drop']*100:+7.2f}% | {row['Accuracy_Without_Feature']*100:10.2f}% | {row['Loss_Increase']:+13.4f} | {row['Max_Correlation_With_Others']:.3f}\n")
            f.write("-" * 85 + "\n\n")

    logger.info(f"UNIFIED PIPELINE COMPLETE! Final report saved to: {report_path}")


if __name__ == "__main__":
    main()
