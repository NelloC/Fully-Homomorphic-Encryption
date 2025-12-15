"""
Concrete ML Regression Pipeline (Compatible Models)

This script performs the following steps:
1. Loads pre-processed patient data (BIS/MAP monitoring).
2. Defines feature sets based on common medical covariates and drug effects.
3. Iterates through various compatible regression models (Linear, ElasticNet, SGD, Decision Tree).
4. Trains the clear-text model, evaluates its performance (MAE/RMSE), and saves parameters.
5. Iterates from 2 to 15 bits, training, compiling, and executing the model using 
   Concrete ML for Homomorphic Encryption (FHE) inference.
6. Stores all clear and FHE predictions/metrics in CSV files.
"""

import time
import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path

# ============================================================
# GENERAL CONFIGURATION
# ============================================================
# Define the output directory for results
OUTPUT_DIR = Path("/Users/aconelli/concrete_ml/LFPAP/LAST_training_LFPAP/results_aki")

# Import Scikit-learn standard libraries for data processing and metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import imbalanced-learn for handling imbalanced datasets (SMOTE)
try:
    from imblearn.combine import SMOTEENN, SMOTETomek
except ImportError:
    print("imbalanced-learn not found. Install with: pip install imbalanced-learn")
    exit()

# Import standard Scikit-learn models
from sklearn.linear_model import LogisticRegression as SkLR
# Note: min_samples_leaf=3 added for better generalization
from sklearn.tree import DecisionTreeClassifier as SkDT
# Note: n_estimators=100 and min_samples_leaf=3 added
from sklearn.ensemble import RandomForestClassifier as SkRF
from sklearn.svm import LinearSVC as SkSVC
import xgboost as xgb

# --- CONCRETE ML (Native Classes) ---
# Import Concrete ML models for Fully Homomorphic Encryption (FHE) support
try:
    from concrete.ml.sklearn import (
        LogisticRegression as ConcreteLR,
        DecisionTreeClassifier as ConcreteDT,
        RandomForestClassifier as ConcreteRF,
        LinearSVC as ConcreteSVC,
        XGBClassifier as ConcreteXGB
    )
    HAS_CONCRETE = True
except ImportError as e:
    print(f"Concrete ML import error: {e}")
    HAS_CONCRETE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define file paths for datasets
AKI_FILE = '/Users/aconelli/concrete_ml/LFPAP/data/aki_7d_from_api.csv'
INTEROP_FILE = '/Users/aconelli/concrete_ml/LFPAP/data/interop_from_api.csv'
LAB_FILE = '/Users/aconelli/concrete_ml/LFPAP/data/lab_for_test_from_api.csv'

# ============================================================
# 1. DATA PREPARATION
# ============================================================

def load_data():
    """
    Loads and merges datasets, handles missing values, and splits data into training and test sets.
    """
    print("Load dataset...")
    try:
        # Load and merge CSV files
        df = pd.read_csv(AKI_FILE)
        df = df.merge(pd.read_csv(INTEROP_FILE), on='caseid', how='inner')
        df = df.merge(pd.read_csv(LAB_FILE), on='caseid', how='inner')
        
        # Process target variable 'AKI'
        df['AKI'] = pd.to_numeric(df['AKI'], errors='coerce')
        df = df.dropna(subset=['AKI'])
        df['AKI'] = df['AKI'].astype(int)
        df = df.drop(columns=['caseid'])
        
        # Impute missing values with the median strategy
        imputer = SimpleImputer(strategy='median')
        X = df.drop(columns=['AKI']).values
        X = imputer.fit_transform(X)
        y = df['AKI'].values
        
        # Split dataset into training and testing sets (80/20 split)
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None 

# ============================================================
# 2. EVALUATION AND SAVING FUNCTIONS
# ============================================================

def get_metrics(y_true, y_pred, time_infer, model_name, type_run, params, bits=None):
    """
    Calculates performance metrics (Accuracy, F1, Precision, Recall).
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    return {
        "Model": model_name,
        "Type": type_run,
        "Bits": bits,
        "Depth": params.get('max_depth'),
        "Accuracy": acc,
        "F1_Score": f1,
        "Precision": prec,
        "Recall": rec,
        "Inference_Time_s": time_infer
    }

def save_results(metrics_dict, y_true, y_pred, y_scores, filename_suffix):
    """
    Saves metrics and predictions to CSV files.
    """
    # Save metrics
    file_m_name = f"metrics_{filename_suffix}.csv"
    file_m_path = OUTPUT_DIR / file_m_name
    
    df_m = pd.DataFrame([metrics_dict])
    hdr = not file_m_path.exists()
    df_m.to_csv(file_m_path, mode='a', index=False, header=hdr)
    
    # Save predictions
    clean_name = metrics_dict['Model'].replace(" ", "_").replace("(", "").replace(")", "")
    file_p_name = f"preds_{clean_name}_{metrics_dict['Type']}"
    
    if 'FHE' in metrics_dict['Type']:
        bits_str = f"_{metrics_dict['Bits']}bits"
        depth_str = f"_D{metrics_dict['Depth']}" if metrics_dict['Depth'] is not None else ""
        file_p_name += f"{bits_str}{depth_str}"
    
    file_p_name += ".csv"
    file_p_path = OUTPUT_DIR / file_p_name
    
    df_p = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_score': y_scores if y_scores is not None else np.zeros_like(y_pred)
    })
    df_p.to_csv(file_p_path, index=False)

# ============================================================
# 3. CLEAR TEXT LOOP (OPTIMIZED)
# ============================================================

def run_clear_text(X_train, X_test, y_train, y_test):
    """
    Runs evaluation for standard Scikit-learn models (non-encrypted).
    Uses SMOTETomek for balancing.
    """
    print("\nSTARTING CLEAR TEXT EVALUATION (Optimized SMOTE)")
    
    # Standard scaler for linear models
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Initialize SMOTETomek with partial sampling strategy
    sm = SMOTETomek(sampling_strategy=0.8, random_state=42)

    # Define models: Name, Instance, Needs Scaling
    models = [
        ("Logistic Regression", SkLR(class_weight='balanced', max_iter=1000), True), 
        ("SVM (Linear)", SkSVC(class_weight='balanced', dual="auto"), True),
        ("Decision Tree", SkDT(class_weight='balanced', min_samples_leaf=3), False),
        ("Random Forest", SkRF(class_weight='balanced', n_estimators=100, min_samples_leaf=3), False),
        ("XGBoost", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), False)
    ]

    depths = range(4, 11)
    
    for name, model_inst, needs_scale in models:
        
        # Select appropriate feature set (scaled or unscaled)
        X_tr_base = X_train_s if needs_scale else X_train
        X_te = X_test_s if needs_scale else X_test
        
        # Apply SMOTETomek resampling to the training set
        X_res, y_res = sm.fit_resample(X_tr_base, y_train) 
        
        # Determine depth iteration
        current_depths = depths if name in ["Decision Tree", "Random Forest", "XGBoost"] else [None]
        
        for d in current_depths:
            display_name = f"{name} (D={d})" if d else name
            
            # Set parameters including max_depth
            if d: 
                model_inst.set_params(max_depth=d)
            
            print(f"    Training {display_name}...", end="\r")
            
            try:
                # Fit model on balanced dataset
                model_inst.fit(X_res, y_res)
                
                # Inference
                t0 = time.time()
                y_pred = model_inst.predict(X_te)
                t_inf = time.time() - t0
                
                # Get prediction scores if available
                if hasattr(model_inst, "predict_proba"):
                    y_scores = model_inst.predict_proba(X_te)[:, 1]
                elif hasattr(model_inst, "decision_function"):
                    y_scores = model_inst.decision_function(X_te)
                else:
                    y_scores = y_pred 
                
                # Calculate and save metrics
                m = get_metrics(y_test, y_pred, t_inf, name, "Clear_Optimized", {"max_depth": d})
                print(f"    {display_name}: Acc={m['Accuracy']:.4f} | F1={m['F1_Score']:.4f} | Prec={m['Precision']:.4f} | Rec={m['Recall']:.4f} | Time={t_inf:.5f}s")
                
                save_results(m, y_test, y_pred, y_scores, "clear_text_optimized")
                
            except Exception as e:
                print(f"    Error {display_name}: {e}")

# ============================================================
# 4. FHE LOOP (OPTIMIZED)
# ============================================================

def run_fhe(X_train, X_test, y_train, y_test):
    """
    Runs evaluation for Concrete ML models (FHE-compatible).
    Uses MinMax scaling and SMOTETomek.
    """
    if not HAS_CONCRETE: return
    print("\nSTARTING FHE EVALUATION (Optimized, extended to 12 bits)")

    # Limit test set size for FHE performance
    TEST_SET_LIMIT = 100

    # MinMax Scaling is mandatory for FHE
    fhe_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_fhe = fhe_scaler.fit_transform(X_train)
    X_test_fhe = fhe_scaler.transform(X_test)

    # Apply test set limit
    if TEST_SET_LIMIT and TEST_SET_LIMIT < len(X_test_fhe):
        X_test_fhe = X_test_fhe[:TEST_SET_LIMIT]
        y_test = y_test[:TEST_SET_LIMIT]

    # Initialize SMOTETomek
    sm = SMOTETomek(sampling_strategy=0.8, random_state=42)
    
    # Apply SMOTETomek to scaled training set
    X_train_fhe_res, y_train_fhe_res = sm.fit_resample(X_train_fhe, y_train) 
    
    # Extended bit range for testing
    bits_list = range(2, 16) # Test 7, 8, 9, 10, 11, 12 bits
    depths = range(4, 11)   
    
    models_fhe = [
        ("Random Forest", ConcreteRF, False), 
        ("XGBoost", ConcreteXGB, False),
        ("SVM (Linear)", ConcreteSVC, True)
    ]
    
    for name, model_cls, needs_linear_setup in models_fhe:
        
        current_depths = depths if name in ["Decision Tree", "Random Forest", "XGBoost"] else [None]
        
        for b in bits_list:
            for d in current_depths:
                display_name = f"{name} FHE (B={b}, D={d})" if d else f"{name} FHE (B={b})"
                print(f"    Processing {display_name}...", end="\r")
                
                try:
                    params = {"n_bits": b}
                    if d: params["max_depth"] = d
                    # Ensure n_estimators matches Scikit-learn RF for consistency
                    if name == "Random Forest": params["n_estimators"] = 100 
                    
                    model = model_cls(**params)
                    
                    # Train on balanced and scaled dataset
                    model.fit(X_train_fhe_res, y_train_fhe_res) 
                    
                    # Compilation
                    start_comp = time.time()
                    model.compile(X_train_fhe_res) 
                    t_comp = time.time() - start_comp
                    
                    # Inference
                    fhe_mode = "execute" 
                    start_inf = time.time()
                    y_pred = model.predict(X_test_fhe, fhe=fhe_mode) 
                    t_inf = time.time() - start_inf
                    
                    y_scores = y_pred
                    
                    m = get_metrics(y_test, y_pred, t_inf, name, f"FHE_Optimized", {"max_depth": d}, bits=b)
                    m["Compile_Time_s"] = t_comp
                    m["FHE_Mode"] = fhe_mode
                    
                    print(f"    {display_name}: Acc={m['Accuracy']:.4f} | F1={m['F1_Score']:.4f} | Prec={m['Precision']:.4f} | Rec={m['Recall']:.4f} | Time={t_inf:.5f}s (Comp Time: {t_comp:.1f}s)")
                    
                    save_results(m, y_test, y_pred, y_scores, "fhe_optimized")

                except Exception as e:
                    print(f"    Error FHE {display_name}: {e}")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    if not OUTPUT_DIR.exists():
        print(f"Creating results directory: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    X_train, X_test, y_train, y_test = load_data()
    
    if X_train is not None:
        # Run optimized Clear Text evaluation
        run_clear_text(X_train, X_test, y_train, y_test)
        
        # Run optimized FHE evaluation
        run_fhe(X_train, X_test, y_train, y_test)
        
        print(f"\nAll tasks completed. Check files in '{OUTPUT_DIR}'.")
    else:
        print("\nExecution interrupted due to data loading error.")
