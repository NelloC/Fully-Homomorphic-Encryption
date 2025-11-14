"""
FHE XGBoost Pipeline with Optuna Hyperparameter Tuning

This script performs the following steps:
1. Loads pre-processed time-series data (from Parquet files).
2. Defines a complex feature set based on signal processing.
3. Uses Optuna to find the best hyperparameters for a standard XGBClassifier
   using k-fold cross-validation.
4. Trains a final clear-text XGBoost model with the best parameters.
5. Evaluates the clear-text model's performance (Accuracy, F1).
6. Iterates from 2 to 15 bits, training and compiling a Concrete ML
   XGBClassifier using the *same best hyperparameters*.
7. Evaluates the FHE model's clear and simulated performance for each bit-width.
"""

from pathlib import Path
import time
import sys
import numpy as np
import pandas as pd

# Data Science & ML Libraries
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

# Handle optional imports
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("Optuna not found. Please install it: pip install optuna")
    sys.exit(1)

try:
    from concrete.ml.sklearn import XGBClassifier as ConcreteXGBClassifier
except ImportError:
    print("Concrete ML not found. Please install it: pip install concrete-ml")
    sys.exit(1)

# -----------------------------
# SETTINGS
# -----------------------------
SIGNAL_FEATURE = ['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2', 'etco2', 'mac', 'pp_ct', 'rf_ct', 'body_temp']
STATIC_FEATURE = ["age", "bmi", "asa"]
HALF_TIME_FILTERING = [60, 3*60, 10*60] # Time windows in seconds

DATASET_FOLDER = Path("data/datasets/30_s_dataset")
RANDOM_SEED = 42

# -----------------------------
# FEATURE NAME DEFINITION
# -----------------------------
FEATURE_NAME = (
    [
        f"{signal}_constant_{half_time}"
        for signal in SIGNAL_FEATURE
        for half_time in HALF_TIME_FILTERING
    ]
    + [
        f"{signal}_slope_{half_time}"
        for signal in SIGNAL_FEATURE
        for half_time in HALF_TIME_FILTERING
    ]
    + [
        f"{signal}_std_{half_time}"
        for signal in SIGNAL_FEATURE
        for half_time in HALF_TIME_FILTERING
    ]
    + STATIC_FEATURE
)
# Remove the 'std' feature for the shortest time window (e.g., 60s)
FEATURE_NAME = [x for x in FEATURE_NAME if f"std_{HALF_TIME_FILTERING[0]}" not in x]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def df_to_numpy(df: pd.DataFrame, features: list):
    """Converts DataFrame to NumPy arrays, dropping NaNs."""
    df_clean = df.dropna(subset=features + ["label"])
    X = df_clean[features].to_numpy(dtype=float)
    y = df_clean["label"].to_numpy(dtype=int)
    return X, y

def objective_xgboost(trial: optuna.Trial, data_train_cv: list, data_test_cv: list, features: list) -> float:
    """
    Optuna objective function for XGBoost hyperparameter tuning
    using cross-validation.
    """
    # Define the hyperparameter search space for the trial
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }

    f1_scores = []
    # Perform cross-validation
    for i in range(len(data_train_cv)):
        train_df = data_train_cv[i]
        val_df = data_test_cv[i]
        
        X_train_cv, y_train_cv = df_to_numpy(train_df, features)
        X_val_cv, y_val_cv = df_to_numpy(val_df, features)
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_cv, y_train_cv)
        y_pred_val = model.predict(X_val_cv)
        
        # We optimize for the F1 score
        f1_scores.append(f1_score(y_val_cv, y_pred_val))
        
    return np.mean(f1_scores)

def main():
    """Main execution pipeline."""
    
    # -----------------------------
    # IMPORT DATA
    # -----------------------------
    print(f"Loading data from {DATASET_FOLDER}...")
    try:
        data = pd.read_parquet(DATASET_FOLDER / 'cases/')
        static = pd.read_parquet(DATASET_FOLDER / 'meta.parquet')
    except FileNotFoundError:
        print(f"Error: Data files not found in {DATASET_FOLDER}.")
        print("Please ensure 'cases/' and 'meta.parquet' exist.")
        sys.exit(1)
        
    data = data.merge(static, on='caseid')

    train = data[data['split'] == "train"]
    test = data[data['split'] == "test"]

    # Convert final train/test sets to NumPy
    X_train, y_train = df_to_numpy(train, FEATURE_NAME)
    X_test, y_test = df_to_numpy(test, FEATURE_NAME)

    print(f"{len(X_train):,d} train samples, {len(X_test):,d} test samples, {y_test.mean():.2%} positive rate.")

    # -----------------------------
    # TRAIN XGBOOST CLEAR MODEL (with Optuna)
    # -----------------------------
    print("\n--- Starting Optuna Hyperparameter Tuning ---")
    
    # Prepare cross-validation folds
    number_fold = len(train.cv_split.unique())
    data_train_cv = [train[train.cv_split != f'cv_{i}'] for i in range(number_fold)]
    data_test_cv = [train[train.cv_split == f'cv_{i}'] for i in range(number_fold)]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective_xgboost(trial, data_train_cv, data_test_cv, FEATURE_NAME),
        n_trials=100, # Number of tuning trials
        show_progress_bar=True,
    )

    best_params = study.best_params
    print(f"Best parameters found by Optuna: {best_params}")

    # --- Fit final clear model ---
    print("\n--- Training Final Clear-Text Model ---")
    model = xgb.XGBClassifier(**best_params, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train, verbose=0) # verbose=0 to silence fit messages

    # -----------------------------
    # EVALUATE CLEAR MODEL
    # -----------------------------
    start = time.time()
    y_pred_clear = model.predict(X_test)
    clear_time = time.time() - start

    acc_clear = accuracy_score(y_test, y_pred_clear)
    f1_clear = f1_score(y_test, y_pred_clear)

    print(f"\n--- Clear Model Evaluation ---")
    print(f"XGB Clear | Accuracy: {acc_clear:.4f}, F1: {f1_clear:.4f}, Inference time: {clear_time:.2f}s")
    
    clear_pred_file = Path("predictions_clear.csv")
    pd.DataFrame({
        "y_true": y_test,
        "y_pred_clear": y_pred_clear
    }).to_csv(clear_pred_file, index=False)
    print(f"Clear predictions saved to {clear_pred_file}")

    # -----------------------------
    # FHE CONCRETE-ML LOOP
    # -----------------------------
    print("\n" + "="*50)
    print("STARTING CONCRETE ML (FHE) EVALUATION")
    print("="*50)

    fhe_results = []

    for n_bits in range(2, 16):
        print(f"\n--- FHE Evaluation: n_bits = {n_bits} ---")

        try:
            # Create and configure the Concrete-ML model with n_bits
            fhe_model = ConcreteXGBClassifier(
                n_estimators=best_params.get("n_estimators", 100),
                max_depth=best_params.get("max_depth", 3),
                learning_rate=best_params.get("learning_rate", 0.1),
                n_bits=n_bits, # Apply quantization bits
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )

            # Train (in clear) and compile for FHE
            fhe_model.fit(X_train, y_train)
            
            print("Compiling FHE circuit...")
            start_compile = time.time()
            fhe_model.compile(X_train)
            compile_time = time.time() - start_compile
            print(f"Compilation completed in {compile_time:.2f}s")

            # 1. Inference in clear (quantized model, no FHE)
            start_clear_q = time.time()
            y_pred_clear_q = fhe_model.predict(X_test)
            clear_q_time = time.time() - start_clear_q
            acc_clear_q = accuracy_score(y_test, y_pred_clear_q)
            f1_clear_q = f1_score(y_test, y_pred_clear_q)
            print(f"Quantized Clear | Accuracy: {acc_clear_q:.4f}, F1: {f1_clear_q:.4f}, Time: {clear_q_time:.2f}s")

            # 2. Inference simulated in FHE
            start_fhe = time.time()
            y_pred_fhe = fhe_model.predict(X_test, fhe="simulate")
            fhe_time = time.time() - start_fhe
            acc_fhe = accuracy_score(y_test, y_pred_fhe)
            f1_fhe = f1_score(y_test, y_pred_fhe)
            print(f"FHE Simulate   | Accuracy: {acc_fhe:.4f}, F1: {f1_fhe:.4f}, Time: {fhe_time:.2f}s")

            # Save predictions for this bit-width
            fhe_pred_file = Path(f"predictions_fhe_{n_bits}bits.csv")
            pd.DataFrame({
                "y_true": y_test,
                "y_pred_fhe_sim": y_pred_fhe
            }).to_csv(fhe_pred_file, index=False)
            print(f"📂 FHE predictions ({n_bits} bits) saved to {fhe_pred_file}")
            
            fhe_results.append({
                "n_bits": n_bits,
                "acc_fhe_sim": acc_fhe,
                "f1_fhe_sim": f1_fhe,
                "compile_time_s": compile_time,
                "inference_time_s": fhe_time,
                "status": "Success"
            })

        except Exception as e:
            print(f" FHE ERROR for n_bits={n_bits}: {e}")
            fhe_results.append({
                "n_bits": n_bits,
                "acc_fhe_sim": 0.0,
                "f1_fhe_sim": 0.0,
                "compile_time_s": 0.0,
                "inference_time_s": 0.0,
                "status": f"Error: {str(e)[:100]}"
            })
    
    # Save final summary of FHE evaluation
    df_fhe_results = pd.DataFrame(fhe_results)
    df_fhe_results.to_csv("fhe_evaluation_summary_optuna.csv", index=False)
    print("\n FHE evaluation summary saved to 'fhe_evaluation_summary_optuna.csv'")

if __name__ == "__main__":
    main()