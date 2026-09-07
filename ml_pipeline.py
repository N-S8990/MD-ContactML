from MDAnalysis.analysis import distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
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
        
        features = np.zeros((n_frames, len(target1_atoms) * len(target2_atoms)))
        
        for frame_index, trajectory_index in enumerate(frame_indices):
            u.trajectory[trajectory_index]
            features[frame_index] = distances.distance_array(
                target1_atoms.positions,
                target2_atoms.positions
            ).ravel()

        return pd.DataFrame(features, columns=feature_names), np.full(n_frames, label)

    def build_dataset(self, topology1, trajectories1, topology2, trajectories2,
                      max_frames=None, frame_stride=1):
        """Build a labeled dataset from two groups of trajectories."""
        feature_frames = []
        labels = []
        groups = []

        for trajectory_index, trajectory in enumerate(trajectories1):
            trajectory_features, trajectory_labels = self._extract_distances(
                topology1, trajectory, label=0, max_frames=max_frames, frame_stride=frame_stride
            )
            feature_frames.append(trajectory_features)
            labels.append(trajectory_labels)
            groups.append(np.full(len(trajectory_labels), f"class0_replica_{trajectory_index}"))

        for trajectory_index, trajectory in enumerate(trajectories2):
            trajectory_features, trajectory_labels = self._extract_distances(
                topology2, trajectory, label=1, max_frames=max_frames, frame_stride=frame_stride
            )
            feature_frames.append(trajectory_features)
            labels.append(trajectory_labels)
            groups.append(np.full(len(trajectory_labels), f"class1_replica_{trajectory_index}"))

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
    
    def __init__(self, n_splits=3, test_fold=0, random_state=42):
        self.n_splits = n_splits
        self.test_fold = test_fold
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def preprocess(self, X, y, groups):
        """Cleans, splits by replica, and scales the dataset without leakage."""
        logger.info("Preprocessing data...")
        
        # 1. Clean data (handle NaN/Inf)
        if isinstance(X, pd.DataFrame):
            X = X.replace([np.inf, -np.inf], np.nan)
            if X.isna().sum().sum() > 0:
                logger.warning(f"Found {X.isna().sum().sum()} missing values. Filling with column means.")
                X = X.fillna(X.mean())
        
        # 2. Hold out entire trajectories. Random frame-level splits would put highly
        # correlated snapshots from one replica in both train and test sets.
        groups = np.asarray(groups)
        y = np.asarray(y)
        class_group_counts = [len(np.unique(groups[y == label])) for label in np.unique(y)]
        if len(class_group_counts) != 2 or min(class_group_counts) < self.n_splits:
            raise ValueError(
                f"Replica-aware evaluation needs at least {self.n_splits} independent "
                f"trajectories per class; found {class_group_counts}."
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
        logger.info(
            "Replica-aware split %s: %s train replicas, %s held-out replicas.",
            self.test_fold,
            len(np.unique(groups[train_idx])), len(np.unique(groups[test_idx]))
        )
        
        # 3. Standardization (fit ONLY on train to avoid data leakage)
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame if input was DataFrame (to keep feature names)
        if isinstance(X, pd.DataFrame):
            X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X.columns)
            X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X.columns)
        else:
            X_train = X_train_scaled
            X_test = X_test_scaled
            
        logger.info(f"Preprocessing complete. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test


# CORRELATION ANALYZING

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """Computes feature correlations and identifies redundant features for removal."""
    
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        
    def find_most_correlated_pair(self, X_train):
        """
        Finds the pair of features with the highest absolute Pearson correlation.
        Returns the pair and their correlation value, or None if max corr < threshold.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a pandas DataFrame to compute correlations with feature names.")
            
        logger.info(f"Computing correlation matrix for {X_train.shape[1]} features...")
        
        # Compute correlation matrix
        corr_matrix = X_train.corr().abs()
        
        # Extract upper triangle without diagonal to find unique pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find the max correlation value
        max_corr = upper_tri.max().max()
        
        if pd.isna(max_corr) or max_corr < self.threshold:
            logger.info(f"No feature pairs found with correlation >= {self.threshold} (max is {max_corr:.4f}).")
            return None, None, max_corr
            
        # Get the feature names for the maximum correlation
        # argmax() flattens the matrix, so we unravel it to get 2D indices
        idx = np.unravel_index(np.nanargmax(upper_tri.values), upper_tri.shape)
        feat_A = upper_tri.index[idx[0]]
        feat_B = upper_tri.columns[idx[1]]
        
        logger.info(f"Highest correlation found: {feat_A} and {feat_B} (corr = {max_corr:.4f})")
        return feat_A, feat_B, max_corr
        
    def get_feature_to_drop(self, X_train, feat_A, feat_B):
        """
        Given a highly correlated pair, decides which one to drop.
        Drops the one that has a higher average absolute correlation with ALL other features.
        """
        # Calculate mean absolute correlation with all other features
        corr_A = X_train.corrwith(X_train[feat_A]).abs().mean()
        corr_B = X_train.corrwith(X_train[feat_B]).abs().mean()
        
        if corr_A > corr_B:
            logger.info(f"Dropping {feat_A} (avg corr: {corr_A:.4f}) over {feat_B} (avg corr: {corr_B:.4f})")
            return feat_A
        else:
            logger.info(f"Dropping {feat_B} (avg corr: {corr_B:.4f}) over {feat_A} (avg corr: {corr_A:.4f})")
            return feat_B


# MODEL TRAIN

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains ML models and records their performance metrics."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Initialize models as per Paper 3
        self.models = {
            'lr': LogisticRegression(
                random_state=random_state, 
                max_iter=1000, 
                solver='lbfgs'
            ),
            'rf': RandomForestClassifier(
                random_state=random_state, 
                n_estimators=100, 
                max_depth=None
            ),
            'mlp': MLPClassifier(
                random_state=random_state,
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                early_stopping=False
            )
        }
        
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
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            if y_prob is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_prob)
                except ValueError:
                    # In case only one class is present in y_test (rare in stratified split)
                    roc_auc = float('nan')
            else:
                roc_auc = float('nan')
                
            metrics[name] = {
                'accuracy': acc,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'recall': recall
            }
            
            logger.debug(f"[{name.upper()}] Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f} | Recall: {recall:.4f}")
            
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
        
    def run(self, X_train, X_test, y_train, y_test, model_to_track='rf'):
        """
        Runs the iterative elimination loop.
        Returns the final feature subset and the elimination log.
        """
        logger.info("Starting Iterative Feature Elimination Loop")
        
        current_X_train = X_train.copy()
        current_X_test = X_test.copy()
        
        # Iteration 0: Baseline
        logger.info(f"--- ITERATION 0 (Baseline) | {current_X_train.shape[1]} features ---")
        metrics = self.trainer.train_and_evaluate(current_X_train, current_X_test, y_train, y_test)
        
        baseline_acc = metrics[model_to_track]['accuracy']
        
        self._record_log(0, "None (Baseline)", current_X_train.shape[1], float('nan'), metrics)
        
        iteration = 1
        
        # Initialize progress bar for the elimination loop
        total_to_drop = current_X_train.shape[1] - self.min_features
        pbar = tqdm(total=total_to_drop, desc="Eliminating Features", unit="feat")
        
        while current_X_train.shape[1] > self.min_features:
            logger.debug(f"--- ITERATION {iteration} | {current_X_train.shape[1]} features ---")
            
            # 1. Find most correlated pair
            feat_A, feat_B, max_corr = self.analyzer.find_most_correlated_pair(current_X_train)
            
            # Check stopping condition 1: No high correlations left
            if feat_A is None:
                logger.info("STOPPING: No highly correlated features remain.")
                break
                
            # 2. Decide which to drop
            feat_to_drop = self.analyzer.get_feature_to_drop(current_X_train, feat_A, feat_B)
            
            # 3. Drop it
            current_X_train = current_X_train.drop(columns=[feat_to_drop])
            current_X_test = current_X_test.drop(columns=[feat_to_drop])
            
            # 4. Retrain and evaluate
            metrics = self.trainer.train_and_evaluate(current_X_train, current_X_test, y_train, y_test)
            current_acc = metrics[model_to_track]['accuracy']
            
            # 5. Log
            self._record_log(iteration, feat_to_drop, current_X_train.shape[1], max_corr, metrics)
            
            # Update progress bar
            pbar.set_postfix({'acc': f"{current_acc:.2f}", 'corr': f"{max_corr:.2f}"})
            pbar.update(1)
            
            # Check stopping condition 2: Accuracy drop too large
            if (baseline_acc - current_acc) > self.accuracy_tolerance:
                logger.warning(f"STOPPING: Accuracy dropped by more than tolerance " 
                               f"({baseline_acc:.4f} -> {current_acc:.4f}). "
                               f"Reverting last drop.")
                # Revert the drop
                current_X_train[feat_to_drop] = X_train[feat_to_drop]
                current_X_test[feat_to_drop] = X_test[feat_to_drop]
                # Remove the last log entry as it was reverted
                self.log.pop()
                break
                
            iteration += 1
            
        pbar.close()
            
        if current_X_train.shape[1] <= self.min_features:
            logger.info(f"STOPPING: Reached minimum feature count ({self.min_features}).")
            
        logger.info(f"Elimination complete. Final feature count: {current_X_train.shape[1]}")
        
        log_df = pd.DataFrame(self.log)
        return current_X_train.columns.tolist(), log_df
        
    def _record_log(self, iteration, dropped_feature, num_features, corr_val, metrics):
        """Helper to structure the log dictionary."""
        entry = {
            'iteration': iteration,
            'dropped_feature': dropped_feature,
            'remaining_features': num_features,
            'dropped_corr': corr_val
        }
        
        # Flatten metrics into the log row
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
        
