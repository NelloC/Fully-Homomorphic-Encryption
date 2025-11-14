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

import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_preprocessing import StandardScaler
from sklearn.base import clone

# Concrete ML imports for FHE compatible models
from concrete.ml.sklearn import LinearRegression as ConcreteLinearRegression
from concrete.ml.sklearn import ElasticNet as ConcreteElasticNet
from concrete.ml.sklearn import SGDRegressor as ConcreteSGDRegressor
from concrete.ml.sklearn import DecisionTreeRegressor as ConcreteDecisionTreeRegressor

# --- Global Definitions ---

REGRESSORS = {
    "LinearRegression": ConcreteLinearRegression(),
    "ElasticNet": ConcreteElasticNet(),
    "SGDRegressor": ConcreteSGDRegressor(max_iter=1000),
    "DecisionTreeRegressor": ConcreteDecisionTreeRegressor(),
}

# Core demographic covariates
COVARIATES = ['age', 'sex', 'height', 'weight']
# Concentration Effect (Ce) Eleveld model features for BIS target
CE_BIS_ELEVELD = ['Ce_Prop_Eleveld', 'Ce_Rem_Eleveld']
# Concentration Effect (Ce) Eleveld model features for MAP target
CE_MAP_ELEVELD = ['Ce_Prop_MAP_Eleveld', 'Ce_Rem_MAP_Eleveld']
# Plasma Concentration (Cp) Eleveld model features
C_PLASMA_ELEVELD = ['Cp_Prop_Eleveld', 'Cp_Rem_Eleveld']


def compute_metrics(df: pd.DataFrame, y_col: str):
    """
    Computes Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
    for a given prediction column against the true values.
    """
    y_true = np.array(df[f"true_{y_col}"])
    y_pred = np.array(df[f"pred_{y_col}"])
    diff = y_pred - y_true

    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")

    return np.argmax(diff), np.argmin(diff)


def define_features(feature_set: str) -> list:
    """Selects the input features based on the specified feature set name."""
    
    if feature_set == 'All':
        X_cols = COVARIATES + ['bmi', 'lbm', 'mean_HR'] + CE_MAP_ELEVELD + CE_BIS_ELEVELD + C_PLASMA_ELEVELD
    elif feature_set == '-bmi':
        X_cols = COVARIATES + ['lbm', 'mean_HR'] + CE_BIS_ELEVELD + CE_MAP_ELEVELD + C_PLASMA_ELEVELD
    elif feature_set == '-lbm':
        X_cols = COVARIATES + ['bmi', 'mean_HR'] + CE_BIS_ELEVELD + CE_MAP_ELEVELD + C_PLASMA_ELEVELD
    elif feature_set == '-hr':
        X_cols = COVARIATES + ['bmi', 'lbm'] + CE_BIS_ELEVELD + CE_MAP_ELEVELD + C_PLASMA_ELEVELD
    elif feature_set == '-Cplasma':
        X_cols = COVARIATES + ['bmi', 'lbm', 'mean_HR'] + CE_BIS_ELEVELD + CE_MAP_ELEVELD
    elif feature_set == '-Cmap':
        X_cols = COVARIATES + ['bmi', 'lbm', 'mean_HR'] + CE_BIS_ELEVELD + C_PLASMA_ELEVELD
    elif feature_set == '-Cbis':
        X_cols = COVARIATES + ['bmi', 'lbm', 'mean_HR'] + CE_MAP_ELEVELD + C_PLASMA_ELEVELD
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
        
    return X_cols


def load_datasets(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads training and testing datasets, with error handling."""
    try:
        train_full = pd.read_csv(train_path, index_col=0)
        test_full = pd.read_csv(test_path, index_col=0)
        return train_full, test_full
    except FileNotFoundError:
        print(f"Error: Required files '{train_path}' or '{test_path}' not found.")
        print("Please ensure data files are in the current working directory.")
        raise


def main():
    """Main execution pipeline for training and FHE evaluation."""
    
    # --- Load Datasets ---
    try:
        patients_train_full, patients_test_full = load_datasets(
            "./Patients_train.csv", "./Patients_test.csv"
        )
    except FileNotFoundError:
        return

    # Create results directory
    os.makedirs('./saved_reg', exist_ok=True)
    
    # ================================
    # Main Loop over Feature Subsets
    # ================================
    
    FEATURE_CONFIGS = ['All', '-bmi', '-lbm', '-hr', '-Cplasma', '-Cmap', '-Cbis']

    for feature_set in FEATURE_CONFIGS:
        
        # Undersampling step
        STEP = 60
        
        # Filter the test data based on availability flags (full_BIS/full_MAP == 0)
        full_test_data_bis = patients_test_full[patients_test_full['full_BIS'] == 0].copy()
        full_test_data_map = patients_test_full[patients_test_full['full_MAP'] == 0].copy()

        # Apply undersampling for current train/test data (using test data as the source for the current experiment setup)
        train_bis_df = full_test_data_bis.iloc[::STEP].copy()
        test_bis_df = full_test_data_bis.iloc[::STEP].copy()
        train_map_df = full_test_data_map.iloc[::STEP].copy()
        test_map_df = full_test_data_map.iloc[::STEP].copy()

        print("\n" + "="*60)
        print(f"Processing Feature Set: {feature_set}")
        print("="*60)

        try:
            X_cols = define_features(feature_set)
        except ValueError as e:
            print(e)
            continue

        # Filter and clean DataFrames: remove rows with NaNs in feature or target columns
        bis_cols = X_cols + ['caseid', 'BIS', 'Time']
        map_cols = X_cols + ['caseid', 'MAP', 'Time'] # Removed 'train_set' as it's not consistently used here

        train_bis_df = train_bis_df.filter(items=bis_cols).dropna(subset=X_cols + ['BIS'])
        test_bis_df = test_bis_df.filter(items=bis_cols).dropna(subset=X_cols + ['BIS'])
        train_map_df = train_map_df.filter(items=map_cols).dropna(subset=X_cols + ['MAP'])
        test_map_df = test_map_df.filter(items=map_cols).dropna(subset=X_cols + ['MAP'])
        
        # Load results or initialize results DataFrames
        try:
            results_bis_df = pd.read_csv("./results_BIS.csv", index_col=0)
            results_map_df = pd.read_csv("./results_MAP.csv", index_col=0)
        except Exception:
            results_bis_df = full_test_data_bis[['Time', 'caseid', 'BIS']].copy()
            results_map_df = full_test_data_map[['Time', 'caseid', 'MAP']].copy()


        # ================================
        # Loop over Regressors
        # ================================
        for regressor_name, regressor_base in REGRESSORS.items():
            
            # File path to save the clear model parameters
            filename = f'./saved_reg/reg_{regressor_name}_feat_{feature_set}_test.pkl'

            # Loop over target variables (BIS and MAP)
            for y_col, train_df, test_df, results_df in [
                ('BIS', train_bis_df, test_bis_df, results_bis_df),
                ('MAP', train_map_df, test_map_df, results_map_df)
            ]:
                if len(train_df) == 0 or len(test_df) == 0:
                    print(f"Skipping {regressor_name} for {y_col}: No valid data points.")
                    continue

                # --- Data Scaling ---
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(train_df[X_cols])
                X_test_scaled = scaler.transform(test_df[X_cols])
                y_train = train_df[y_col].values

                # --- Training (Clear Model) ---
                start_train = time.time()
                try:
                    # Clone the base regressor instance
                    rg_clear = clone(regressor_base)  
                except TypeError as e:
                    print(f"Skipping {regressor_name}: Cloning failed due to {e}")
                    continue
                    
                rg_clear.fit(X_train_scaled, y_train)
                t_train = time.time() - start_train
                print(f"\n-- {regressor_name} for {y_col} trained ({t_train:.3f}s) --")

                # --- Safe and Lightweight Model Saving (Parameters only) ---
                params_to_save = {
                    'name': regressor_name,
                    'class': type(rg_clear).__name__,
                    'hyperparams': rg_clear.get_params(),
                    'weights': getattr(rg_clear, 'coef_', None),
                    'intercept': getattr(rg_clear, 'intercept_', None),
                }
                with open(filename, 'wb') as f:
                    pickle.dump(params_to_save, f)

                # --- Clear Model Prediction and Evaluation ---
                start_clear = time.time()
                y_pred_clear = rg_clear.predict(X_test_scaled)
                t_clear = time.time() - start_clear

                print(f"→ {regressor_name} {y_col} clear inference time: {t_clear:.3f}s")
                
                # Store clear prediction results
                col_name_clear = f'pred_{y_col}_{regressor_name}_clear'
                y_pred_series = pd.Series(y_pred_clear, index=test_df.index)
                results_df.loc[y_pred_series.index, col_name_clear] = y_pred_series

                # Prepare temporary DataFrame for clear metrics calculation
                temp_test_data = pd.DataFrame()
                temp_test_data['caseid'] = test_df['caseid']  
                temp_test_data[f'true_{y_col}'] = test_df[y_col]
                temp_test_data[f'pred_{y_col}'] = y_pred_clear

                print(f"\nTest Results for {y_col} (Clear Model):")
                compute_metrics(temp_test_data, y_col)

                # ================================
                # FHE Compilation and Prediction Loop
                # ================================
                
                # Iterate over different bit depths for quantization (from 2 to 15 bits)
                for n_bits in range(2, 16):
                    print(f"\n--- {regressor_name} {y_col} with n_bits={n_bits} ---")
                    
                    # Instantiate Concrete ML model with the current n_bits
                    if regressor_name == "LinearRegression":
                        rg_fhe = ConcreteLinearRegression(n_bits=n_bits)
                    elif regressor_name == "ElasticNet":
                        # Note: ElasticNet uses max_iter=1000 by default in Concrete ML
                        rg_fhe = ConcreteElasticNet(n_bits=n_bits) 
                    elif regressor_name == "SGDRegressor":
                        rg_fhe = ConcreteSGDRegressor(n_bits=n_bits, max_iter=1000)
                    elif regressor_name == "DecisionTreeRegressor":
                        rg_fhe = ConcreteDecisionTreeRegressor(n_bits=n_bits)
                    else:
                        continue 

                    try:
                        # 1. Train the FHE-enabled model
                        rg_fhe.fit(X_train_scaled, y_train)
                        
                        # 2. Compile the model for FHE
                        print("Compiling model for FHE...")
                        rg_fhe.compile(X_train_scaled)

                        # 3. FHE Inference (Execute the compiled circuit)
                        start_fhe = time.time()
                        # Use fhe="simulate" or fhe="execute" based on environment. 
                        # We use "execute" as per your original request.
                        y_pred_fhe = rg_fhe.predict(X_test_scaled, fhe="execute") 
                        t_fhe = time.time() - start_fhe

                        print(f"→ {regressor_name} {y_col} FHE inference time: {t_fhe:.3f}s")

                        # Store FHE prediction results
                        col_name_fhe = f'pred_{y_col}_{regressor_name}_fhe_{n_bits}bit'
                        results_df[col_name_fhe] = pd.Series(y_pred_fhe, index=test_df.index)

                        # Check prediction loss due to quantization/FHE
                        mae_fhe_vs_clear = np.mean(np.abs(y_pred_fhe - y_pred_clear))
                        print(f"MAE (FHE vs Clear) for {n_bits} bits: {mae_fhe_vs_clear:.4f}")

                    except Exception as e:
                        # Handle compilation or FHE execution failures
                        print(f"[FHE Execution Failed] {regressor_name} {y_col} with n_bits={n_bits}: {e}")
                        # Mark result as NaN if FHE fails
                        results_df[f'pred_{y_col}_{regressor_name}_fhe_{n_bits}bit'] = np.nan


        # Final Results Saving after processing all regressors for this feature set
        results_bis_df.to_csv("./results_BIS.csv")
        results_map_df.to_csv("./results_MAP.csv")


if __name__ == '__main__':
    main()