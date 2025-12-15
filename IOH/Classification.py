import time
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
import warnings

# Imbalanced-learn library
try:
    from imblearn.combine import SMOTETomek
except ImportError:
    print("imbalanced-learn not found. Install with: pip install imbalanced-learn")
    exit(1)

# Scikit-learn Imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb

# Concrete ML Imports
try:
    from concrete.ml.sklearn import (
        XGBClassifier as ConcreteXGB,
        LogisticRegression as ConcreteLR,
        DecisionTreeClassifier as ConcreteDT,
        RandomForestClassifier as ConcreteRF,
        LinearSVC as ConcreteSVC
    )
except ImportError:
    print("Concrete ML not found. FHE steps will be skipped.")
    exit(1)

# Configuration
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# -----------------------------
# SETTINGS
# -----------------------------
SIGNAL_FEATURE = ['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2', 'etco2', 'mac', 'pp_ct', 'rf_ct', 'body_temp']
STATIC_FEATURE = ["age", "bmi", "asa"]
HALF_TIME_FILTERING = [60, 3*60, 10*60]

# Paths
dataset_folder = Path("/Users/aconelli/concrete_ml/bobauin/data/datasets/30_s_dataset")
results_folder = Path("/Users/aconelli/concrete_ml/bobauin/last_train_hypo/results_hypo/")
results_folder.mkdir(parents=True, exist_ok=True)

# -----------------------------
# IMPORT DATA
# -----------------------------
print("Loading Dataset...")
try:
    data = pd.read_parquet(dataset_folder / 'cases/')
    static = pd.read_parquet(dataset_folder / 'meta.parquet')
    data = data.merge(static, on='caseid')

    train = data[data['split'] == "train"]
    test = data[data['split'] == "test"]
except Exception as e:
    print(f"Error loading data: {e}")
    print("Ensure the path 'data/datasets/30_s_dataset' is correct.")
    exit(1)

# -----------------------------
# FEATURE NAMES
# -----------------------------
FEATURE_NAME = (
    [f"{signal}_constant_{half_time}" for signal in SIGNAL_FEATURE for half_time in HALF_TIME_FILTERING] +
    [f"{signal}_slope_{half_time}" for signal in SIGNAL_FEATURE for half_time in HALF_TIME_FILTERING] +
    [f"{signal}_std_{half_time}" for signal in SIGNAL_FEATURE for half_time in HALF_TIME_FILTERING] +
    STATIC_FEATURE
)
# Exclude specific std feature based on filtering logic
FEATURE_NAME = [x for x in FEATURE_NAME if f"std_{HALF_TIME_FILTERING[0]}" not in x]

# -----------------------------
# UTILS
# -----------------------------
def df_to_numpy(df, features):
    """Converts DataFrame to numpy arrays for X and y, dropping NaNs."""
    df_clean = df.dropna(subset=features + ["label"])
    X = df_clean[features].to_numpy(dtype=float)
    y = df_clean["label"].to_numpy(dtype=int)
    return X, y

# Convert data
X_train, y_train = df_to_numpy(train, FEATURE_NAME)
X_test, y_test = df_to_numpy(test, FEATURE_NAME)

print(f"{len(X_train):,d} train samples, {len(X_test):,d} test samples.")

# -----------------------------
# DATA BALANCING (SMOTETomek)
# -----------------------------
print("Applying SMOTETomek to training set for balancing...")
try:
    smote_tomek = SMOTETomek(random_state=42)
    # X_train_sm and y_train_sm will be used ONLY for final model training
    X_train_sm, y_train_sm = smote_tomek.fit_resample(X_train, y_train)
    print(f"Final Training Dataset size (Balanced): {len(X_train_sm):,d} samples.")
except Exception as e:
    print(f"Error applying SMOTETomek: {e}")
    # Fallback to original dataset if resampling fails
    X_train_sm, y_train_sm = X_train, y_train
    print("Proceeding with the original (unbalanced) training dataset.")

# -----------------------------
# MODEL AND OPTUNA CONFIGURATION
# -----------------------------

def objective(trial, model_name, X_train, y_train, cv_splits):
    """
    Generic Optuna objective function.
    Uses original X_train/y_train for Cross-Validation to avoid leakage from synthetic SMOTE data.
    """
    params = {}
    model = None

    # Define Search Space for each model
    if model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 50),        
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_jobs": -1
        }
        # Calculate positive weight for this fold
        pos_weight = (y_train[cv_splits[0][0]] == 0).sum() / (y_train[cv_splits[0][0]] == 1).sum()
        model = xgb.XGBClassifier(**params, scale_pos_weight=pos_weight)

    elif model_name == "LogisticRegression":
        params = {
            "C": trial.suggest_float("C", 0.001, 10.0, log=True),
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000
        }
        model = LogisticRegression(**params)

    elif model_name == "DecisionTree":
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "class_weight": "balanced"
        }
        model = DecisionTreeClassifier(**params)

    elif model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 50),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "class_weight": "balanced",
            "n_jobs": -1
        }
        model = RandomForestClassifier(**params)

    elif model_name == "LinearSVC":
        params = {
            "C": trial.suggest_float("C", 0.001, 10.0, log=True),
            "class_weight": "balanced",
            "dual": "auto",
            "max_iter": 2000
        }
        model = LinearSVC(**params)

    # Manual Cross-Validation loop
    scores = []
    for train_idx, val_idx in cv_splits:
        X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # XGBoost: Update scale_pos_weight for current fold
        if model_name == "XGBoost":
             pos_weight_fold = (y_tr_fold == 0).sum() / (y_tr_fold == 1).sum()
             model = xgb.XGBClassifier(**params, scale_pos_weight=pos_weight_fold)
            
        model.fit(X_tr_fold, y_tr_fold)
        preds = model.predict(X_val_fold)
        scores.append(f1_score(y_val_fold, preds, zero_division=0))

    return np.mean(scores)

# Mappings
CONCRETE_MAP = {
    "XGBoost": ConcreteXGB,
    "LogisticRegression": ConcreteLR,
    "DecisionTree": ConcreteDT,
    "RandomForest": ConcreteRF,
    "LinearSVC": ConcreteSVC
}

STANDARD_MAP = {
    "XGBoost": xgb.XGBClassifier,
    "LogisticRegression": LogisticRegression,
    "DecisionTree": DecisionTreeClassifier,
    "RandomForest": RandomForestClassifier,
    "LinearSVC": LinearSVC
}

# Prepare CV indices for Optuna
cv_indices = []
try:
    for i in range(len(train.cv_split.unique())):
        train_idx = np.where(train.cv_split != f'cv_{i}')[0]
        val_idx = np.where(train.cv_split == f'cv_{i}')[0]
        cv_indices.append((train_idx, val_idx))
except AttributeError:
    print("Warning: 'cv_split' column not found. Using default CV strategy if needed.")

TEST_SET_LIMIT = 500

# -----------------------------
# MAIN PROCESSING LOOP
# -----------------------------
all_clear_metrics = []

for model_name in STANDARD_MAP.keys():
    print(f"\n=============================================")
    print(f"PROCESSING MODEL: {model_name}")
    print(f"=============================================")

    # 1. OPTIMIZATION (Optuna)
    print(f"--- Tuning Hyperparameters (Optuna) ---")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    study.optimize(
        lambda trial: objective(trial, model_name, X_train, y_train, cv_indices),
        n_trials=20, 
        show_progress_bar=True
    )
    
    best_params = study.best_params
    
    # Add fixed final parameters
    if model_name == "LogisticRegression": best_params.update({"class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000})
    if model_name == "LinearSVC": best_params.update({"class_weight": "balanced", "dual": "auto"})
    if model_name in ["DecisionTree", "RandomForest"]: best_params.update({"class_weight": "balanced"})
    if model_name == "XGBoost":
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        best_params['scale_pos_weight'] = pos_weight

    print(f"Best Params: {best_params}")

    # 2. TRAIN CLEAR MODEL
    # Using the Standard Library implementation fitted on Balanced (SMOTE) Data
    print(f"--- Training Clear Text Model (Fitted on SMOTETomek data) ---")
    clear_model_cls = STANDARD_MAP[model_name]
    
    # Optional: Clean params for specific models if needed
    final_params = best_params.copy()
    if model_name in ["DecisionTree", "RandomForest", "LogisticRegression", "LinearSVC"]:
        if 'class_weight' in final_params: del final_params['class_weight']
    if model_name == "XGBoost":
        if 'scale_pos_weight' in final_params: del final_params['scale_pos_weight']

    clear_model = clear_model_cls(**final_params)
    
    # Fit on Balanced Data
    clear_model.fit(X_train_sm, y_train_sm)
    
    # 3. EVALUATE CLEAR MODEL ON TRAINING SET
    print(f"--- Evaluating on Training Set (Clear Text) ---")
    
    start = time.time()
    # Predict on original training data
    y_pred_train_clear = clear_model.predict(X_train)
    clear_time = time.time() - start
    
    acc_clear = accuracy_score(y_train, y_pred_train_clear)
    f1_clear = f1_score(y_train, y_pred_train_clear, zero_division=0)
    prec_clear = precision_score(y_train, y_pred_train_clear, zero_division=0)
    rec_clear = recall_score(y_train, y_pred_train_clear, zero_division=0)    
    
    print(f"Clear Training Results -> Acc: {acc_clear:.4f}, F1: {f1_clear:.4f}, Prec: {prec_clear:.4f}, Rec: {rec_clear:.4f}, Time: {clear_time:.4f}s")

    # Save predictions
    pd.DataFrame({"y_true": y_train, "y_pred": y_pred_train_clear}).to_csv(
        results_folder / f"pred_clear_train_{model_name}_SMOTETomek.csv", index=False
    )
    
    clear_metrics_data = {
        "Model": model_name,
        "Type": "Clear_Train",
        "Accuracy": acc_clear,
        "F1": f1_clear,
        "Precision": prec_clear,
        "Recall": rec_clear,
        "Time": clear_time
    }
    
    pd.DataFrame([clear_metrics_data]).to_csv(
        results_folder / f"metrics_clear_train_{model_name}_SMOTETomek.csv", index=False
    )
    print(f"   Saved training metrics: metrics_clear_train_{model_name}_SMOTETomek.csv")
    
    all_clear_metrics.append(clear_metrics_data)

    # 4. FHE EVALUATION (Concrete ML)
    print(f"--- Starting FHE Evaluation (Concrete ML) ---")
    
    if model_name not in CONCRETE_MAP:
        print(f"Skipping FHE for {model_name} (Not implemented in map)")
        continue

    concrete_cls = CONCRETE_MAP[model_name]
    bits_to_test = [2, 4, 6, 8, 10, 12, 14, 16]
    
    # Limit test set for FHE
    X_test_fhe = X_test[:TEST_SET_LIMIT]
    y_test_fhe = y_test[:TEST_SET_LIMIT]

    for n_bits in bits_to_test:
        print(f"   > Processing FHE with n_bits={n_bits}...")
        
        try:
            fhe_params = best_params.copy()
            fhe_params["n_bits"] = n_bits
            
            # Clean params for Concrete ML
            if "n_jobs" in fhe_params: del fhe_params["n_jobs"]
            if "class_weight" in fhe_params: del fhe_params["class_weight"]
            if "scale_pos_weight" in fhe_params: del fhe_params["scale_pos_weight"]

            # Instantiate and Train
            model_fhe = concrete_cls(**fhe_params)
            model_fhe.fit(X_train_sm, y_train_sm)
            
            # Compile (using original train data for calibration)
            model_fhe.compile(X_train)
            
            # Execute Inference
            start_fhe = time.time()
            y_pred_fhe = model_fhe.predict(X_test_fhe, fhe="execute")
            fhe_time = time.time() - start_fhe
            
            # Metrics on Test Set
            acc_fhe = accuracy_score(y_test_fhe, y_pred_fhe)
            f1_fhe = f1_score(y_test_fhe, y_pred_fhe, zero_division=0)
            prec_fhe = precision_score(y_test_fhe, y_pred_fhe, zero_division=0)
            rec_fhe = recall_score(y_test_fhe, y_pred_fhe, zero_division=0)
            
            print(f"     [FHE {n_bits}b] Acc: {acc_fhe:.4f}, F1: {f1_fhe:.4f}, Time: {fhe_time:.4f}s")
            
            fhe_metrics_data = {
                "Model": model_name,
                "Type": "FHE",
                "Bits": n_bits,
                "Accuracy": acc_fhe,
                "F1": f1_fhe,
                "Precision": prec_fhe,
                "Recall": rec_fhe,
                "Time": fhe_time
            }
            
            # Save metrics and predictions
            pd.DataFrame([fhe_metrics_data]).to_csv(
                results_folder / f"metrics_fhe_{model_name}_{n_bits}bits.csv", index=False
            )
            pd.DataFrame({"y_true": y_test_fhe, "y_pred": y_pred_fhe}).to_csv(
                results_folder / f"pred_fhe_{model_name}_{n_bits}bits.csv", index=False
            )

        except Exception as e:
            print(f"     Error processing FHE {n_bits} bits: {e}")

print("\nProcessing completed.")
