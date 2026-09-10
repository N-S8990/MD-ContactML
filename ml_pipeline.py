from MDAnalysis.analysis import distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, balanced_accuracy_score, precision_score
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import MDAnalysis as mda
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import concurrent.futures

try:
    from cuml.ensemble import RandomForestClassifier as GPU_RF  # type: ignore
    HAS_CUML = True
except Exception as e:
    GPU_RF = RandomForestClassifier
    HAS_CUML = False


# FEATURE EXTRACTION

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MDFeatureExtractor:
    """Extracts pairwise C-alpha distances from MD trajectories to build ML features."""
    
    def __init__(self, 
                 target1_selection="chainID A ", 
                 target2_selection="chainID B ",
                 target1_name="GH",
                 target2_name="GHR"):
        self.target1_selection = target1_selection
        self.target2_selection = target2_selection
        self.target1_name = target1_name
        self.target2_name = target2_name
        
    def _extract_distances(self, topology, trajectory, label, max_frames=None, frame_stride=1):
        """Extracts distances for a single trajectory."""
        logger.info(f"Loading trajectory: {trajectory}")
        
        # Load Universe
        u = mda.Universe(topology, trajectory)
        
        # Select atom groups
        target1_atoms = u.select_atoms(self.target1_selection)
        target2_atoms = u.select_atoms(self.target2_selection)
        
        if len(target1_atoms) == 0 or len(target2_atoms) == 0:
            raise ValueError(f"Selection returned 0 atoms. Check selections: {self.target1_selection}, {self.target2_selection}")
            
        logger.info(f"Selected {len(target1_atoms)} {self.target1_name} atoms and {len(target2_atoms)} {self.target2_name} atoms.")
        
        # Prepare feature names
        feature_names = []
        for r_atom in target1_atoms:
            for a_atom in target2_atoms:
                # Chain and residue identifiers are stable across independently
                # simulated replicas, unlike atom indices after a mutation.
                feature_names.append(
                    f"{self.target1_name}_{r_atom.chainID}:{r_atom.residue.resid}_"
                    f"{self.target2_name}_{a_atom.chainID}:{a_atom.residue.resid}"
                )
                
        # Iterate through trajectory
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1.")
        frame_indices = list(range(0, len(u.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        n_frames = len(frame_indices)
        if n_frames == 0:
            raise ValueError(f"Trajectory {trajectory} contains no frames to extract.")
        logger.info(f"Extracting features across {n_frames} frames...")
        
        features = np.zeros((n_frames, len(target1_atoms) * len(target2_atoms)), dtype=np.float32)
        
        for frame_index, trajectory_index in enumerate(frame_indices):
            u.trajectory[trajectory_index]
            features[frame_index] = distances.distance_array(
                target1_atoms.positions,
                target2_atoms.positions
            ).ravel()

        return pd.DataFrame(features, columns=feature_names), np.full(n_frames, label)

    def build_dataset(self, topology1, trajectories1, topology2, trajectories2,
                      max_frames=None, frame_stride=1, workers=1):
        """Build a labeled dataset from two groups of trajectories."""
        feature_frames = []
        labels = []
        groups = []

        tasks = []
        # Create a list of all jobs
        for i, traj in enumerate(trajectories1):
            tasks.append((topology1, traj, 0, max_frames, frame_stride, f"class0_replica_{i}"))
        for i, traj in enumerate(trajectories2):
            tasks.append((topology2, traj, 1, max_frames, frame_stride, f"class1_replica_{i}"))

        if workers > 1:
            logger.info(f"Extracting features using {workers} workers...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = []
                for top, traj, label, mf, fs, group_name in tasks:
                    futures.append(
                        executor.submit(self._extract_distances, top, traj, label, mf, fs)
                    )
                
                for future, task in zip(futures, tasks):
                    traj_features, traj_labels = future.result()
                    group_name = task[5]
                    feature_frames.append(traj_features)
                    labels.append(traj_labels)
                    groups.append(np.full(len(traj_labels), group_name))
        else:
            logger.info("Extracting features sequentially...")
            for top, traj, label, mf, fs, group_name in tasks:
                traj_features, traj_labels = self._extract_distances(top, traj, label, mf, fs)
                feature_frames.append(traj_features)
                labels.append(traj_labels)
                groups.append(np.full(len(traj_labels), group_name))

        if not feature_frames:
            raise ValueError("At least one trajectory is required for each dataset group.")

        reference_columns = feature_frames[0].columns
        mismatched = [
            index for index, frame in enumerate(feature_frames[1:], start=1)
            if not frame.columns.equals(reference_columns)
        ]
        if mismatched:
            raise ValueError(
                "Replica topologies do not produce the same contact-feature schema. "
                "Use equivalent chain/residue numbering and selections for every replica; "
                f"mismatched datasets: {mismatched}."
            )

        return (pd.concat(feature_frames, ignore_index=True), np.concatenate(labels),
                np.concatenate(groups))


# PREPROCESSING

logger = logging.getLogger(__name__)

class MDPreprocessor:
    """Preprocesses MD feature data for ML models."""
    
    def __init__(self, n_splits=3, test_fold=0, random_state=42, initial_features=None):
        self.n_splits = n_splits
        self.test_fold = test_fold
        self.random_state = random_state
        self.initial_features = initial_features
        self.scaler = StandardScaler()
        
    def preprocess(self, X, y, groups):
        """Cleans, splits by replica, and scales the dataset without leakage."""
        logger.info("Preprocessing data...")
        
        # 1. Clean data and cast to float32
        if isinstance(X, pd.DataFrame):
            X = X.replace([np.inf, -np.inf], np.nan)
            if X.isna().sum().sum() > 0:
                logger.warning(f"Found {X.isna().sum().sum()} missing values. Filling with column means.")
                X = X.fillna(X.mean())
            X = X.astype(np.float32)
        
        # 2. Hold out entire trajectories or do frame split
        groups = np.asarray(groups)
        y = np.asarray(y)
        
        if self.n_splits == 1:
            logger.warning("cv_folds=1 detected. Using standard 80/20 random frame split. (Note: Frame splitting can cause data leakage in time-series MD data)")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, groups, test_size=0.2, stratify=y, random_state=self.random_state)
        else:
            class_group_counts = [len(np.unique(groups[y == label])) for label in np.unique(y)]
            if len(class_group_counts) != 2 or min(class_group_counts) < self.n_splits:
                raise ValueError(
                    f"Replica-aware evaluation needs at least {self.n_splits} independent "
                    f"trajectories per class; found {class_group_counts}. To bypass, use --cv_folds 1"
                )
            if not 0 <= self.test_fold < self.n_splits:
                raise ValueError(f"test_fold must be between 0 and {self.n_splits - 1}.")
            splitter = StratifiedGroupKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            splits = splitter.split(X, y, groups)
            for fold_index, (train_idx, test_idx) in enumerate(splits):
                if fold_index == self.test_fold:
                    break
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train, groups_test = groups[train_idx], groups[test_idx]
            logger.info(
                "Replica-aware split %s: %s train replicas, %s held-out replicas.",
                self.test_fold,
                len(np.unique(groups_train)), len(np.unique(groups_test))
            )
            
        # 3. Optional Initial Feature Pre-Filter (Based ONLY on X_train)
        if self.initial_features is not None and self.initial_features < X_train.shape[1]:
            logger.info(f"Applying initial pre-filter to keep {self.initial_features} closest contacts...")
            mean_distances = X_train.mean(axis=0)
            closest_cols = mean_distances.nsmallest(self.initial_features).index
            X_train = X_train[closest_cols]
            X_test = X_test[closest_cols]
            logger.info(f"Reduced features from {X.shape[1]} to {X_train.shape[1]}")
        
        # 4. Standardization (fit ONLY on train to avoid data leakage)
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame if input was DataFrame (to keep feature names)
        if isinstance(X, pd.DataFrame):
            X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
            X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
        else:
            X_train = X_train_scaled
            X_test = X_test_scaled
            
        logger.info(f"Preprocessing complete. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test, groups_train, groups_test


# CORRELATION ANALYZING

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """Computes feature correlations and identifies redundant features for removal."""
    
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.corr_matrix = None
        self.mean_corrs = None
        
    def fit(self, X_train):
        logger.info(f"Computing full correlation matrix for {X_train.shape[1]} features once...")
        self.corr_matrix = X_train.corr().abs()
        self.mean_corrs = self.corr_matrix.mean()
        
    def get_features_to_drop_batch(self, active_features, batch_size=50):
        """
        Finds a batch of features to drop that are highly correlated among the ACTIVE features.
        """
        if self.corr_matrix is None:
            raise ValueError("Must call fit() before getting features to drop.")
            
        # Subset the precomputed matrix to only active features
        current_corr = self.corr_matrix.loc[active_features, active_features]
        upper_tri = current_corr.where(np.triu(np.ones(current_corr.shape), k=1).astype(bool))
        
        high_corr_mask = upper_tri >= self.threshold
        if not high_corr_mask.any().any():
            return [], 0.0
            
        pairs = np.argwhere(high_corr_mask.values)
        corrs = [upper_tri.iloc[r, c] for r, c in pairs]
        sorted_indices = np.argsort(corrs)[::-1]
        
        features_to_drop = set()
        
        max_corr_found = corrs[sorted_indices[0]]
        
        for idx in sorted_indices:
            r, c = pairs[idx]
            feat_A = upper_tri.index[r]
            feat_B = upper_tri.columns[c]
            
            if feat_A not in features_to_drop and feat_B not in features_to_drop:
                # Use the global mean correlation to break ties
                if self.mean_corrs[feat_A] > self.mean_corrs[feat_B]:
                    features_to_drop.add(feat_A)
                else:
                    features_to_drop.add(feat_B)
                    
            if len(features_to_drop) >= batch_size:
                break
                
        return list(features_to_drop), max_corr_found


# MODEL TRAIN

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains ML models and records their performance metrics."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        rf_kwargs = {'random_state': random_state, 'n_estimators': 300, 'max_depth': None, 'max_features': 'sqrt'}
        if HAS_CUML:
            logger.info("Initializing GPU-accelerated RandomForestClassifier (cuML)")
            # cuML specific params or defaults
        else:
            logger.info("Initializing CPU-based RandomForestClassifier (sklearn)")
            rf_kwargs['n_jobs'] = -1

        self.models = {
            'lr': LogisticRegression(
                random_state=random_state, 
                max_iter=1000, 
                solver='lbfgs',
                n_jobs=-1,
                C=1.0 # Will be tuned
            ),
            'rf': GPU_RF(**rf_kwargs),
            'mlp': MLPClassifier(
                random_state=random_state,
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                learning_rate_init=0.001,
                early_stopping=True
            )
        }
        
    def tune_hyperparameters(self, X_train, y_train, groups=None, cv_folds=3):
        """Perform a small hyperparameter search for RF and LR using training data only."""
        logger.info("Starting hyperparameter tuning on training data...")
        
        cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state) if cv_folds > 1 else 3
        
        # Tune RF
        rf_param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [None, 30, 60],
            'max_features': ['sqrt', 0.5]
        }
        logger.info("Tuning Random Forest...")
        rf_search = RandomizedSearchCV(
            self.models['rf'], rf_param_grid, n_iter=5, cv=cv, 
            scoring='accuracy', random_state=self.random_state, 
            n_jobs=1 if HAS_CUML else -1
        )
        # cuML RF may not support groups in fit, but RandomizedSearchCV handles it by passing groups to cv.split()
        rf_search.fit(X_train, y_train, groups=groups)
        logger.info(f"Best RF params: {rf_search.best_params_} (Val Acc: {rf_search.best_score_:.4f})")
        self.models['rf'] = rf_search.best_estimator_
        
        # Tune LR
        lr_param_grid = {'C': [0.1, 1.0, 10.0]}
        logger.info("Tuning Logistic Regression...")
        lr_search = RandomizedSearchCV(
            self.models['lr'], lr_param_grid, n_iter=3, cv=cv, 
            scoring='accuracy', random_state=self.random_state, n_jobs=-1
        )
        lr_search.fit(X_train, y_train, groups=groups)
        logger.info(f"Best LR params: {lr_search.best_params_} (Val Acc: {lr_search.best_score_:.4f})")
        self.models['lr'] = lr_search.best_estimator_
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, models_to_run=None):
        """
        Trains specified models and returns a dictionary of metrics.
        If models_to_run is None, runs all available models.
        """
        if models_to_run is None:
            models_to_run = list(self.models.keys())
            
        metrics = {}
        
        for name in models_to_run:
            if name not in self.models:
                logger.warning(f"Model '{name}' not found. Skipping.")
                continue
                
            model = self.models[name]
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Evaluate
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            
            if y_prob is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_prob)
                except ValueError:
                    roc_auc = float('nan')
            else:
                roc_auc = float('nan')
                
            metrics[name] = {
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'precision': precision,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'recall': recall
            }
            
            logger.debug(f"[{name.upper()}] Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")
            
        return metrics


# ELIMINATION LOOP

logger = logging.getLogger(__name__)

class FeatureEliminationLoop:
    """Orchestrates the iterative removal of highly correlated features."""
    
    def __init__(self, corr_threshold=0.90, accuracy_tolerance=0.05, min_features=10):
        self.analyzer = CorrelationAnalyzer(threshold=corr_threshold)
        self.trainer = ModelTrainer()
        self.accuracy_tolerance = accuracy_tolerance
        self.min_features = min_features
        self.log = []
        
    def run(self, X_train, y_train, groups_train, X_test, y_test, model_to_track='rf', cv_folds=3):
        """
        Runs the iterative elimination loop using an internal validation split.
        Returns the final feature subset and the elimination log.
        """
        logger.info("Starting Iterative Feature Elimination Loop")
        
        # Create an internal validation set from the training data to prevent test leakage
        if cv_folds > 1:
            val_splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            splits = list(val_splitter.split(X_train, y_train, groups_train))
            train_idx, val_idx = splits[0]
            
            X_train_inner = X_train.iloc[train_idx]
            y_train_inner = y_train[train_idx]
            X_val_inner = X_train.iloc[val_idx]
            y_val_inner = y_train[val_idx]
        else:
            from sklearn.model_selection import train_test_split
            X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )
        
        active_features = X_train.columns.tolist()
        
        # Precompute the correlation matrix once on the FULL training set
        self.analyzer.fit(X_train)
        
        # Iteration 0: Baseline (Train tracking model on inner train, eval on inner val)
        logger.info(f"--- ITERATION 0 (Baseline) | {len(active_features)} features ---")
        metrics = self.trainer.train_and_evaluate(X_train_inner, X_val_inner, y_train_inner, y_val_inner, models_to_run=[model_to_track])
        
        baseline_acc = metrics[model_to_track]['accuracy']
        self._record_log(0, "None (Baseline)", len(active_features), float('nan'), metrics)
        
        iteration = 1
        total_to_drop = len(active_features) - self.min_features
        pbar = tqdm(total=total_to_drop, desc="Eliminating Features", unit="feat")
        
        batch_size = 50 # Drop up to 50 features at a time to vastly speed up execution
        
        while len(active_features) > self.min_features:
            logger.debug(f"--- ITERATION {iteration} | {len(active_features)} features ---")
            
            # 1. Find batch of correlated features to drop using precomputed matrix
            features_to_drop, max_corr = self.analyzer.get_features_to_drop_batch(active_features, batch_size=batch_size)
            
            if not features_to_drop:
                logger.info("STOPPING: No highly correlated features remain.")
                break
                
            # 2. Update active features
            active_features = [f for f in active_features if f not in features_to_drop]
            
            # 3. Retrain ONLY the tracking model on validation set
            metrics = self.trainer.train_and_evaluate(X_train_inner[active_features], X_val_inner[active_features], y_train_inner, y_val_inner, models_to_run=[model_to_track])
            current_acc = metrics[model_to_track]['accuracy']
            
            # 4. Log
            self._record_log(iteration, features_to_drop[0] + f" (+{len(features_to_drop)-1} more)", len(active_features), max_corr, metrics)
            
            pbar.set_postfix({'acc': f"{current_acc:.2f}", 'corr': f"{max_corr:.2f}"})
            pbar.update(len(features_to_drop))
            
            if (baseline_acc - current_acc) > self.accuracy_tolerance:
                logger.warning(f"STOPPING: Accuracy dropped by more than tolerance. Reverting last batch.")
                # Restore active features while preserving original column order
                reverted_set = set(active_features + features_to_drop)
                active_features = [f for f in X_train.columns if f in reverted_set]
                self.log.pop()
                break
                
            iteration += 1
            
        pbar.close()
        
        logger.info(f"Elimination complete. Final feature count: {len(active_features)}")
        
        # Train ALL models one final time on the FULL optimized feature set training data, evaluated on untouched Test set
        logger.info(f"--- FINAL EVALUATION (ON TEST SET) | {len(active_features)} features ---")
        final_metrics = self.trainer.train_and_evaluate(X_train[active_features], X_test[active_features], y_train, y_test)
        self._record_log(iteration, "FINAL_TEST_EVAL", len(active_features), float('nan'), final_metrics)
        
        log_df = pd.DataFrame(self.log)
        return active_features, log_df
        
    def _record_log(self, iteration, dropped_feature, num_features, corr_val, metrics):
        entry = {
            'iteration': iteration,
            'dropped_feature': dropped_feature,
            'remaining_features': num_features,
            'dropped_corr': corr_val
        }
        for model_name, model_metrics in metrics.items():
            for metric_name, val in model_metrics.items():
                entry[f"{model_name}_{metric_name}"] = val
        self.log.append(entry)


# VISUALIZER.PY 

logger = logging.getLogger(__name__)

class PipelineVisualizer:
    """Generates plots and visualizations for the feature elimination pipeline."""
    
    def __init__(self, output_dir="results/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Set style
        sns.set_theme(style="whitegrid")
        
    def plot_accuracy_vs_features(self, log_df, models=['lr', 'rf', 'mlp']):
        """Plots the accuracy curve as features are removed."""
        plt.figure(figsize=(12, 6))
        
        for model in models:
            col_name = f"{model}_accuracy"
            if col_name in log_df.columns:
                plt.plot(log_df['remaining_features'], log_df[col_name], 
                         marker='o', markersize=4, label=model.upper())
                
        # Invert x-axis so it reads left-to-right as features are removed
        plt.gca().invert_xaxis()
        
        plt.title('Model Accuracy vs. Number of Remaining Features')
        plt.xlabel('Number of Features Remaining')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, 'accuracy_vs_features.png')
        plt.savefig(out_path, dpi=300)
        logger.info(f"Saved plot to {out_path}")
        plt.close()
        
    def plot_correlation_heatmap(self, X, title, filename):
        """Plots a correlation heatmap for a given feature matrix."""
        # If too many features, it's unreadable, so limit it
        if X.shape[1] > 100:
            logger.warning(f"Too many features ({X.shape[1]}) for a clean heatmap. " 
                           "Plotting correlation matrix without annotations.")
            
        plt.figure(figsize=(10, 8))
        corr = X.corr()
        
        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        lw = 0.5 if X.shape[1] <= 100 else 0
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                    square=True, linewidths=lw, cbar_kws={"shrink": .5},
                    xticklabels=False, yticklabels=False)
                    
        plt.title(title)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300)
        logger.info(f"Saved heatmap to {out_path}")
        plt.close()
        
