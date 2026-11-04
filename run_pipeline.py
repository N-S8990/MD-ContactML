import argparse
import importlib.metadata
import json
import logging
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from ml_pipeline import (
    ComprehensiveFeatureExtractor,
    MDPreprocessor,
    FeatureEliminationLoop,
    ModelTrainer,
    LOFOAblationEngine,
    GroupLOFOEngine,
    PipelineVisualizer,
    categorize_cluster,
    build_grouped_features,
    grouped_permutation_importance
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
    parser.add_argument("--test_replica", type=int, default=5,
                        help="Replica number reserved for final testing (default: 5).")
    parser.add_argument("--corr_threshold", type=float, default=0.98,
                        help="Pearson correlation threshold to eliminate redundant features (default: 0.98).")
    parser.add_argument("--acc_tolerance", type=float, default=0.05,
                        help="Maximum accuracy drop allowed during elimination before stopping (default: 0.05).")
    parser.add_argument("--skip_lofo", action="store_false", dest="run_lofo",
                        help="Skip the LOFO ablation study to save runtime.")
    parser.add_argument("--skip_cluster_lofo", action="store_false", dest="run_cluster_lofo",
                        help="Skip the Group LOFO cluster ablation study.")
    parser.add_argument("--lofo_min_drop", type=float, default=0.01,
                        help="Minimum accuracy drop threshold to list ablated features in report/plot (default: 0.01 = 1%%).")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max frames to extract per trajectory (optional, for testing).")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Stride between trajectory frames (default: 1).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel trajectory extraction workers (default: 1).")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable cached trajectory feature extraction.")
    parser.add_argument("--cv_folds", type=int, default=4,
                        help="Group-aware cross-validation folds on training replicas (default: 4).")
    parser.add_argument("--bootstrap_samples", type=int, default=200,
                        help="Bootstrap samples for held-out metric confidence intervals (default: 200).")
    parser.add_argument("--permutation_repeats", type=int, default=10,
                        help="Permutation-importance repeats; 0 disables it (default: 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for models, bootstrap intervals, and metadata (default: 42).")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Directory to save all results, figures, and reports (default: results).")

    args = parser.parse_args()

    if len(args.class0_traj) != len(args.class1_traj):
        parser.error("--class0_traj and --class1_traj must contain the same number of replicas.")
    if len(args.class0_traj) < 5:
        parser.error("At least five replicas per condition are required for trajectory-level evaluation.")
    if not 1 <= args.test_replica <= len(args.class0_traj):
        parser.error("--test_replica must identify one of the supplied replicas.")

    input_paths = [args.class0_top, args.class1_top, *args.class0_traj, *args.class1_traj]
    missing_paths = [path for path in input_paths if not Path(path).is_file()]
    if missing_paths:
        parser.error("Input file(s) not found: " + ", ".join(missing_paths))

    if not 0.0 < args.corr_threshold <= 1.0:
        parser.error("--corr_threshold must be greater than 0 and at most 1.")
    if args.acc_tolerance < 0.0:
        parser.error("--acc_tolerance must be non-negative.")
    if args.lofo_min_drop < 0.0:
        parser.error("--lofo_min_drop must be non-negative.")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max_frames must be positive when provided.")
    if args.frame_stride < 1:
        parser.error("--frame_stride must be positive.")
    if args.workers < 1:
        parser.error("--workers must be positive.")
    if args.cv_folds < 2 or args.cv_folds > len(args.class0_traj) - 1:
        parser.error(f"--cv_folds must be between 2 and {len(args.class0_traj) - 1} for the training replicas.")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap_samples must be positive.")
    if args.permutation_repeats < 0:
        parser.error("--permutation_repeats must be non-negative.")

    return args


def write_run_metadata(args, out_dir):
    package_names = ["numpy", "pandas", "scikit-learn", "MDAnalysis", "networkx", "matplotlib", "seaborn"]
    package_versions = {}
    for package_name in package_names:
        try:
            package_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package_name] = "unavailable"

    input_paths = [args.class0_top, args.class1_top, *args.class0_traj, *args.class1_traj]
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed": args.seed,
        "arguments": vars(args),
        "input_files": [
            {"path": str(Path(path).resolve()), "size_bytes": Path(path).stat().st_size,
             "modified_ns": Path(path).stat().st_mtime_ns}
            for path in input_paths
        ],
        "package_versions": package_versions,
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def main():
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    start_time = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(args, out_dir)
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
        frame_stride=args.frame_stride,
        cache_dir=None if args.no_cache else out_dir / "cache",
        workers=args.workers,
    )
    extraction_time = time.time() - t0_ext
    logger.info(f"Feature Extraction Completed in {extraction_time:.2f}s.")
    X_full.to_csv(out_dir / "raw_features.csv", index=False)

    preprocessor = MDPreprocessor()
    test_group = f"rep_{args.test_replica}"
    X_train_scaled, y_train, groups_train, X_test_scaled, y_test = preprocessor.fit_transform_split(
        X_full, y_full, groups_full, test_group=test_group
    )
    logger.info(f"Train Set: {X_train_scaled.shape[0]} frames | Test Set ({test_group}): {X_test_scaled.shape[0]} frames")

    logger.info("Computing training-only Pearson Correlation Matrix...")
    train_corr = X_train_scaled.corr().abs()
    train_corr.to_csv(out_dir / "train_correlation_matrix.csv")
    visualizer.plot_correlation_heatmap(
        train_corr,
        f"Training Correlation Matrix ({X_train_scaled.shape[1]} Features)",
        "train_correlation_heatmap.png"
    )

    t0_elim = time.time()
    eliminator = FeatureEliminationLoop(
        corr_threshold=args.corr_threshold,
        accuracy_tolerance=args.acc_tolerance,
        out_dir=out_dir,
        random_state=args.seed
    )
    active_features, log_df, corr_init, corr_final = eliminator.run(
        X_train_scaled, y_train, groups_train
    )
    elimination_time = time.time() - t0_elim

    visualizer.plot_correlation_heatmap(corr_init, "Initial Feature Correlation Matrix", "corr_matrix_initial.png")
    visualizer.plot_correlation_heatmap(corr_final, "Final Feature Correlation Matrix (Collinearity Removed)", "corr_matrix_final.png")
    visualizer.plot_elimination_trajectory(log_df, "elimination_trajectory.png")

    # The model uses one standardized biological feature per group. This keeps
    # sister representations (distance, soft contact, inverse distance, etc.)
    # from competing with one another.
    X_tr_final = build_grouped_features(X_train_scaled[active_features])
    X_te_final = build_grouped_features(X_test_scaled[active_features])
    grouped_corr = X_tr_final.corr().abs()
    grouped_corr.to_csv(out_dir / "grouped_feature_correlation_matrix.csv")
    grouped_max_corr = pd.Series(np.maximum(
        grouped_corr.where(np.triu(np.ones(grouped_corr.shape), k=1).astype(bool)).max(axis=0).fillna(0),
        grouped_corr.where(np.triu(np.ones(grouped_corr.shape), k=1).astype(bool)).max(axis=1).fillna(0),
    ), index=grouped_corr.columns)

    logger.info(f"Benchmarking grouped classifiers on held-out trajectory group {test_group}...")
    trainer = ModelTrainer(random_state=args.seed)

    benchmark_df = trainer.train_and_evaluate(X_tr_final, y_train, X_te_final, y_test)
    benchmark_df.to_csv(out_dir / "model_benchmark_metrics.csv")

    cv_df = trainer.group_cross_validate(
        X_tr_final, y_train, groups_train,
        n_splits=args.cv_folds
    )
    cv_df.to_csv(out_dir / "group_cv_metrics.csv", index=False)
    uncertainty_df = trainer.bootstrap_test_metrics(
        y_test, n_bootstrap=args.bootstrap_samples, random_state=args.seed
    )
    uncertainty_df.to_csv(out_dir / "test_metric_confidence_intervals.csv", index=False)

    rf = trainer.models["Random Forest"]
    lr = trainer.models["Logistic Regression"]
    rf_importances = rf.feature_importances_
    lr_coefs = lr.coef_[0]

    # Use replica 1 -> replica 2 for feature discovery. The final test replica
    # remains reserved for the benchmark above.
    inner_train_mask = groups_train == "rep_1"
    inner_eval_mask = groups_train == "rep_2"
    inner_rf = RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1)
    inner_rf.fit(X_tr_final.loc[inner_train_mask], y_train[inner_train_mask])
    X_inner_eval = X_tr_final.loc[inner_eval_mask]
    y_inner_eval = y_train[inner_eval_mask]

    logger.info("Computing validation-replica permutation feature importance...")
    if args.permutation_repeats:
        perm = permutation_importance(
            inner_rf, X_inner_eval, y_inner_eval,
            n_repeats=args.permutation_repeats,
            random_state=args.seed,
            n_jobs=args.workers,
        )
        perm_importances = perm.importances_mean
    else:
        logger.info("Skipping permutation importance because --permutation_repeats=0.")
        perm_importances = np.zeros(len(X_tr_final.columns))

    grouped_perm_df = grouped_permutation_importance(
        inner_rf, X_inner_eval, y_inner_eval,
        repeats=max(args.permutation_repeats, 1), random_state=args.seed
    )
    grouped_perm_df.to_csv(out_dir / "grouped_permutation_importance.csv", index=False)

    # Physical class separation is reported independently of model importance.
    effect_records = []
    X_effect = X_tr_final
    for feature in X_effect.columns:
        class0 = X_effect.loc[y_train == 0, feature].to_numpy()
        class1 = X_effect.loc[y_train == 1, feature].to_numpy()
        pooled_sd = np.sqrt((np.var(class0, ddof=1) + np.var(class1, ddof=1)) / 2.0)
        effect_records.append({
            "Feature": feature,
            "Category": categorize_cluster(feature),
            "Class0_Mean": float(np.mean(class0)),
            "Class1_Mean": float(np.mean(class1)),
            "Class1_Minus_Class0": float(np.mean(class1) - np.mean(class0)),
            "Standardized_Mean_Difference": float((np.mean(class1) - np.mean(class0)) / (pooled_sd + 1e-12)),
        })
    pd.DataFrame(effect_records).sort_values(
        "Standardized_Mean_Difference", key=lambda s: s.abs(), ascending=False
    ).to_csv(out_dir / "feature_distribution_effects.csv", index=False)

    prominent_records = []
    for i, feat in enumerate(X_tr_final.columns):
        cat = categorize_cluster(feat)
        prominent_records.append({
            "Feature": feat,
            "Category": cat,
            "RF_Importance": rf_importances[i],
            "Permutation_Importance": perm_importances[i],
            "LR_Coefficient": lr_coefs[i],
            "Shift_Direction_at_Class1": "See distribution effects"
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
        group_engine = GroupLOFOEngine(out_dir=out_dir, random_state=args.seed)
        cluster_df, cluster_stats = group_engine.run(
            X_tr_final.loc[inner_train_mask], y_train[inner_train_mask],
            X_inner_eval, y_inner_eval, prom_df=prom_df
        )
        visualizer.plot_cluster_ablation(cluster_df, "cluster_ablation_accuracy_drop.png", min_drop=args.lofo_min_drop)
        logger.info(f"Group LOFO Completed in {time.time() - t0_cl:.2f}s.")

    lofo_df = None
    lofo_stats = None
    if args.run_lofo:
        t0_lofo = time.time()
        logger.info("Running Leave-One-Feature-Out (LOFO) ablation testing across all features...")
        lofo_engine = LOFOAblationEngine(out_dir=out_dir, random_state=args.seed)
        lofo_df, lofo_stats = lofo_engine.run(
            X_tr_final.loc[inner_train_mask], y_train[inner_train_mask],
            X_inner_eval, y_inner_eval, max_corr_series=grouped_max_corr
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
        f.write(f"Grouped Model Features:            {len(X_tr_final.columns)}\n")
        f.write(f"Held-Out Test Replica Group:       {test_group}\n")
        f.write(f"Elimination Steps:                {len(log_df) - 1}\n\n")
        f.write("Feature discovery note: permutation importance and LOFO/group ablations use the inner training-replica validation split; the final test replica is reserved for benchmarking.\n")
        f.write("Confidence-interval note: test metrics use contiguous block bootstrap to reduce time-series autocorrelation bias.\n\n")

        f.write("-" * 85 + "\n")
        f.write("MODEL BENCHMARK PERFORMANCE (ON HELD-OUT TEST REPLICA)\n")
        f.write("-" * 85 + "\n")
        f.write(benchmark_df.to_string() + "\n\n")

        f.write("-" * 85 + "\n")
        f.write("GROUP-AWARE CROSS-VALIDATION ON TRAINING REPLICAS\n")
        f.write("-" * 85 + "\n")
        f.write(cv_df.to_string(index=False) + "\n\n")

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
            for _, row in drop_clusters.iterrows():
                f.write(f"{row['Rank']:4d} | {row['Cluster'][:30]:30s} | {row['Category'][:28]:28s} | {row['Accuracy_Drop']*100:+7.2f}% | {row['Accuracy_Without_Cluster']*100:10.2f}% | {row['Master_Impact_Score']:13.4f} | {row['Num_Features']:5d}\n")
            f.write("-" * 85 + "\n\n")

        if lofo_df is not None:
            f.write("-" * 85 + "\n")
            f.write("COMPLETE GROUPED FEATURES RANKED BY ACCURACY DROP\n")
            f.write("-" * 85 + "\n")
            f.write(f"LOFO Baseline Accuracy: {lofo_stats['baseline_accuracy'] * 100:.2f}%\n")
            f.write("Positive values mean performance decreased when the grouped feature was removed.\n")
            f.write("Rank | Grouped Feature                 | Category                     | Acc Drop | F1 Drop | AUC Drop | Loss Increase\n")
            f.write("-" * 85 + "\n")
            for _, row in lofo_df.iterrows():
                f.write(f"{row['Rank']:4d} | {row['Feature'][:32]:32s} | {row['Category'][:28]:28s} | {row['Accuracy_Drop']*100:+7.2f}% | {row['F1_Drop']*100:+7.2f}% | {row['AUC_Drop']*100:+8.2f}% | {row['Loss_Increase']:+13.4f}\n")
            f.write("-" * 85 + "\n\n")

            drop_feats = lofo_df[lofo_df["Accuracy_Drop"] > args.lofo_min_drop]
            f.write("-" * 85 + "\n")
            f.write(f"GROUPED FEATURES WITH LOFO ACCURACY DROP > {args.lofo_min_drop*100:.2f}%\n")
            f.write("-" * 85 + "\n")
            f.write(f"LOFO Baseline Accuracy: {lofo_stats['baseline_accuracy'] * 100:.2f}%\n")
            f.write("Rank | Feature Name                   | Category                     | Acc Drop | Acc Without | Loss Increase | Max Corr\n")
            f.write("-" * 85 + "\n")
            for _, row in drop_feats.iterrows():
                f.write(f"{row['Rank']:4d} | {row['Feature'][:30]:30s} | {row['Category'][:28]:28s} | {row['Accuracy_Drop']*100:+7.2f}% | {row['Accuracy_Without_Feature']*100:10.2f}% | {row['Loss_Increase']:+13.4f} | {row['Max_Correlation_With_Others']:.3f}\n")
            f.write("-" * 85 + "\n\n")

    logger.info(f"UNIFIED PIPELINE COMPLETE! Final report saved to: {report_path}")


if __name__ == "__main__":
    main()
