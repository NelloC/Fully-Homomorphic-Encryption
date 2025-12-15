import os
import time
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================
# CUDA/GPU ACCELERATION DETECTION
# ================================
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    DEVICE_INFO = f"GPU ({torch.cuda.get_device_name(0)})" if HAS_CUDA else "CPU"
    print(f"{' CUDA environment detected! Accelerator: ' + DEVICE_INFO if HAS_CUDA else ' No CUDA. Using CPU.'}")
except ImportError:
    HAS_CUDA = False
    DEVICE_INFO = "CPU"
    print("Warning: PyTorch not found. CPU will be used.")

XGB_TREE_METHOD = 'gpu_hist' if HAS_CUDA else 'hist'

# ================================
# CONCRETE ML CONFIGURATION
# ================================
MODELS_CONFIG = {}
HAS_CONCRETE = False

print("\nVerifying Concrete ML installation...")
try:
    import concrete.ml.sklearn as cmls
    HAS_CONCRETE = True
    
    # Base parameters for XGBoost
    xgb_params_base = {"n_estimators": 30, "max_depth": 3, "tree_method": XGB_TREE_METHOD}
    
    # List of models to test
    candidates = [
        ("LinearRegression", "LinearRegression", {}),
        ("Ridge", "Ridge", {"alpha": 1.0}),
        ("Lasso", "Lasso", {"alpha": 0.1}),
        ("ElasticNet", "ElasticNet", {"alpha": 0.1, "l1_ratio": 0.5}),
        ("SGDRegressor", "SGDRegressor", {"max_iter": 2000, "tol": 1e-3}),
        ("LinearSVR", "LinearSVR", {"C": 1.0, "max_iter": 2000}),
        ("DecisionTree", "DecisionTreeRegressor", {"max_depth": 4}),
        ("RandomForest", "RandomForestRegressor", {"n_estimators": 10, "max_depth": 4}),
        ("XGBoost", "XGBRegressor", xgb_params_base),
        ("KNeighbors", "KNeighborsRegressor", {"n_neighbors": 5})
    ]
    
    # Register available models
    for friendly_name, class_name, params in candidates:
        if hasattr(cmls, class_name):
            model_cls = getattr(cmls, class_name)
            MODELS_CONFIG[friendly_name] = (model_cls, params)
        else:
            print(f"     {class_name} not available. Skipping.")
except ImportError as e:
    print(f"Concrete ML not installed: {e}")

warnings.filterwarnings('ignore')

# ================================
# EXECUTION CONFIGURATION
# ================================
OUTPUT_FOLDER = "/Users/aconelli/concrete_ml/Pharma_regression/last_train_reg/results_regression_FINAL"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TEST_SET_LIMIT = 1000 
BITS_RANGE = range(2, 16)
DEPTH_RANGE = range(4, 11)
METRICS_FILE = os.path.join(OUTPUT_FOLDER, "bis_map_results.csv")

# ================================
# UTILITY FUNCTIONS
# ================================

def calculate_metrics(y_true, y_pred, exec_time_total, model_name, scenario, bits, feature_set, target, eval_set_name):
    """
    Calculates regression metrics and returns a dictionary.
    """
    n_samples = len(y_true)
    time_per_sample = exec_time_total / n_samples if n_samples > 0 else 0
    return {
        "Model": model_name,
        "Target": target,
        "Feature_Set": feature_set,
        "Type": scenario,
        "Eval_Set": eval_set_name, # Distinguishes between Train and Test sets
        "Bits": bits,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred) if n_samples > 1 else 0,
        "Total_Time_s": exec_time_total,
        "Time_Per_Sample_s": time_per_sample,
        "Samples_Evaluated": n_samples
    }

def save_predictions(y_true, y_pred, model_name, target, feature_set, scenario, bits):
    """
    Saves true vs predicted values to a CSV file.
    """
    filename = f"pred_{target}_{feature_set}_{model_name}_{scenario}_{bits}bits.csv".replace(" ", "_")
    path = os.path.join(OUTPUT_FOLDER, filename)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(path, index=False)

def update_metrics_file(metrics_list):
    """
    Overwrites the metrics CSV file with the current list of results.
    Uses atomic writing (write to temp -> rename) to prevent corruption.
    """
    try:
        df_partial = pd.DataFrame(metrics_list)
        cols_order = ["Model", "Type", "Eval_Set", "Bits", "MAE", "RMSE", "R2",
                      "Total_Time_s", "Time_Per_Sample_s", "Samples_Evaluated",
                      "Target", "Feature_Set"]
        
        # Keep only columns that exist
        final_cols = [c for c in cols_order if c in df_partial.columns]
        df_partial = df_partial[final_cols]
        
        tmp_path = METRICS_FILE + ".tmp"
        df_partial.to_csv(tmp_path, index=False)
        os.replace(tmp_path, METRICS_FILE)
    except Exception as e:
        print(f"[Warning] Could not save metrics: {e}")

def add_time_features(df, target_col, n_lags=3):
    """
    Adds Lag and Delta features to the dataframe.
    Correctly calculates deltas based on past lags to avoid data leakage.
    """
    df_new = df.copy()
    
    # 1. Create Lags
    for lag in range(1, n_lags+1):
        # Use ffill/bfill to handle initial NaNs
        df_new[f"{target_col}_lag{lag}"] = df_new[target_col].shift(lag).ffill().bfill()
        
    # 2. Create Deltas (Lag_i - Lag_{i+1})
    for lag in range(1, n_lags):
        current_lag = f"{target_col}_lag{lag}"
        prev_lag = f"{target_col}_lag{lag+1}"
        df_new[f"{target_col}_delta_{lag}_{lag+1}"] = df_new[current_lag] - df_new[prev_lag]
        
    df_new.fillna(0, inplace=True) 
    return df_new

# ================================
# DATA LOADING
# ================================
print("\nLoading Dataset...")
try:
    Patients_train_full = pd.read_csv("/Users/aconelli/concrete_ml/Pharma_regression/Patients_train.csv", index_col=0)
    Patients_test_full = pd.read_csv("/Users/aconelli/concrete_ml/Pharma_regression/Patients_test.csv", index_col=0)
except FileNotFoundError:
    print("Files not found. Generating dummy data for testing.")
    cols = ['age', 'sex', 'height', 'weight', 'bmi', 'lbm', 'mean_HR',
            'Ce_Prop_Eleveld', 'Ce_Rem_Eleveld', 'Ce_Prop_MAP_Eleveld', 'Ce_Rem_MAP_Eleveld',
            'Cp_Prop_Eleveld', 'Cp_Rem_Eleveld', 'BIS', 'MAP', 'full_BIS', 'full_MAP']
    Patients_train_full = pd.DataFrame(np.random.rand(50, len(cols)), columns=cols)
    Patients_test_full = pd.DataFrame(np.random.rand(20, len(cols)), columns=cols)
    Patients_train_full[['full_BIS', 'full_MAP']] = 0
    Patients_test_full[['full_BIS', 'full_MAP']] = 0
    
metrics_list = []

# ================================
# MAIN LOOP
# ================================
feature_sets_to_test = ['All']

for feature_set in feature_sets_to_test:
    print(f"\nFeature Set: {feature_set}")

    # Prepare datasets copies
    train_bis = Patients_train_full.copy()
    train_map = Patients_train_full.copy()
    full_test_bis = Patients_test_full.copy()
    full_test_map = Patients_test_full.copy()

    # Define base features
    base_features = [c for c in Patients_train_full.columns if c not in ['BIS','MAP','full_BIS','full_MAP','caseid','Time','train_set']]

    # Add Time Features (Lags/Deltas)
    train_bis = add_time_features(train_bis, 'BIS')
    full_test_bis = add_time_features(full_test_bis, 'BIS')
    train_map = add_time_features(train_map, 'MAP')
    full_test_map = add_time_features(full_test_map, 'MAP')
    
    # Collect all feature columns
    X_cols = base_features + [c for c in train_bis.columns if 'lag' in c or 'delta' in c]
    X_cols = list(set(X_cols)) # Remove duplicates

    targets = {'BIS': (train_bis, full_test_bis), 'MAP': (train_map, full_test_map)}

    for target, (df_tr, df_te) in targets.items():
        
        valid_cols = [c for c in X_cols if c in df_tr.columns and c in df_te.columns]
        if len(df_tr) == 0 or len(df_te) == 0: continue

        # Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit on Train
        X_train = scaler.fit_transform(df_tr[valid_cols])
        y_train = df_tr[target].values
        
        # Transform on Test (limited if configured)
        if TEST_SET_LIMIT:
            df_te_limited = df_te.iloc[:TEST_SET_LIMIT]
        else:
            df_te_limited = df_te
            
        X_test = scaler.transform(df_te_limited[valid_cols])
        y_test = df_te_limited[target].values

        print(f"      Target: {target} | Train Size: {len(X_train)} | Test Size: {len(X_test)}")

        for model_name, (concrete_class, params) in MODELS_CONFIG.items():
            # ===========================
            # 1. Cleartext Baseline (ON TRAIN SET)
            # ===========================
            try:
                # Use native XGBoost class for clear text baseline if applicable
                if model_name == "XGBoost":
                     model_clear_cls = xgb.XGBRegressor
                else:
                    model_clear_cls = concrete_class 
                    
                model_clear = model_clear_cls(**params)
                t0 = time.time()
                
                # Fit on Train
                model_clear.fit(X_train, y_train)
                
                # --- MODIFIED: Predict on TRAIN SET ---
                y_pred_clear = model_clear.predict(X_train)
                t_clear = time.time() - t0
                
                # Calculate metrics on Training data
                m = calculate_metrics(y_train, y_pred_clear, t_clear, model_name, "Clear_Train", 0, feature_set, target, "Train")
                metrics_list.append(m)
                
                # IMMEDIATE SAVE
                update_metrics_file(metrics_list)
                
                print(f"       Clear (Train): MAE={m['MAE']:.3f} | RMSE={m['RMSE']:.3f} | R2={m['R2']:.3f}")
            except Exception as e:
                print(f"        Clear Error: {e}")
                continue

            if not HAS_CONCRETE: continue

            # ===========================
            # 2. FHE Execution (ON TEST SET)
            # ===========================
            concrete_model_cls = concrete_class
            # Iterate depth only for tree-based models
            depths_to_cycle = DEPTH_RANGE if model_name in ["DecisionTree", "RandomForest", "XGBoost"] else [None]

            for bits in BITS_RANGE:
                for d in depths_to_cycle:
                    display_name = f"FHE ({bits}b)"
                    try:
                        fhe_params = params.copy()
                        fhe_params['n_bits'] = bits
                        if 'n_jobs' in fhe_params: del fhe_params['n_jobs']
                        
                        if d is not None:
                            fhe_params['max_depth'] = d
                            display_name = f"FHE ({bits}b, D={d})"
                        elif 'max_depth' in fhe_params:
                            del fhe_params['max_depth']

                        # Train on Train set
                        model_fhe = concrete_model_cls(**fhe_params)
                        model_fhe.fit(X_train, y_train)
                        
                        # Compile
                        model_fhe.compile(X_train)
                        
                        # Execute on Test set (Limited samples)
                        t_start = time.time()
                        y_pred_fhe = model_fhe.predict(X_test, fhe="execute")
                        t_total = time.time() - t_start
                        
                        scenario_name = display_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_").replace("=", "")
                        
                        m_fhe = calculate_metrics(y_test, y_pred_fhe, t_total, model_name, scenario_name, bits, feature_set, target, "Test")
                        metrics_list.append(m_fhe)
                        
                        # IMMEDIATE SAVE
                        update_metrics_file(metrics_list)
                        
                        save_predictions(y_test, y_pred_fhe, model_name, target, feature_set, scenario_name, bits)
                        print(f"            Done {display_name}: MAE={m_fhe['MAE']:.3f} | R2={m_fhe['R2']:.3f}")
                        
                    except Exception as e:
                        print(f"            Error {display_name}: {e}")

print(f"\nProcessing completed. Final file: {METRICS_FILE}")
