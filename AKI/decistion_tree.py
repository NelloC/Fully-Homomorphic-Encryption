import time
import pandas as pd
import numpy as np
import pickle
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split

# Import Concrete ML components
try:
    from concrete.ml.sklearn import DecisionTreeClassifier as ConcreteDT
    from concrete.ml.quantization import QuantizedModule
except ImportError:
    print("Concrete ML not found. Please install it to run FHE evaluation.")
    sys.exit(1)

# --- CONFIGURATION ---
# Assumed CSV file paths generated from prior steps (data acquisition/preparation)
AKI_FILE = 'aki_7d_from_api.csv'
INTEROP_FILE = 'interop_from_api.csv'
LAB_FILE = 'lab_for_test_from_api.csv'

# Path to save and load the trained model
MODEL_PATH = './decision_tree_model_fhe.pkl'

# --- CORE FUNCTIONS ---

def load_and_merge_data():
    """Loads and merges data from all required CSV files, performs cleaning,
    and splits into training and testing sets.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("Loading and merging data...")
    try:
        df_aki = pd.read_csv(AKI_FILE)
        df_interop = pd.read_csv(INTEROP_FILE)
        df_lab = pd.read_csv(LAB_FILE)
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e.filename}")
        sys.exit(1)

    # 1. Merge dataframes on 'caseid'
    df_merged = df_aki.merge(df_interop, on='caseid', how='inner')
    df_merged = df_merged.merge(df_lab, on='caseid', how='inner')

    # 2. Robust cleaning and final preparation
    # Convert 'AKI' column to numeric, coercing errors (e.g., 'null' strings) to NaN
    df_merged['AKI'] = pd.to_numeric(df_merged['AKI'], errors='coerce')
    # Drop rows where AKI status is unknown (NaN)
    df_merged = df_merged.dropna(subset=['AKI'])
    df_merged['AKI'] = df_merged['AKI'].astype(int)

    # Drop the case identifier and handle missing feature values (NaNs)
    df_merged = df_merged.drop(columns=['caseid'])
    # Fill remaining NaNs (from features) with -1, assuming -1 indicates missing lab values
    df_merged = df_merged.fillna(-1)

    # Separate features (X) and target (y)
    X = df_merged.drop(columns=['AKI']).values
    y = df_merged['AKI'].values

    # Split into training and test sets (80/20 split, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, model_type="Clear Text"):
    """Calculates and prints performance metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n--- Metrics ({model_type}) ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return f1

def train_and_save_dt(X_train, y_train):
    """Trains a standard (clear text) Decision Tree Classifier and saves it."""
    print("\n--- Training Clear Text Model (Scikit-learn) ---")

    # Use parameters compatible with both standard DT and Concrete ML Decision Trees
    model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=4,
        class_weight='balanced',
        min_samples_split=50,
        random_state=12
    )

    model.fit(X_train, y_train)

    # Save the standard model using pickle
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)
    print(f"Trained DT model saved to {MODEL_PATH}")
    return model

def predict_clear_text(model, X_test, y_test):
    """Evaluates the standard (non-FHE) model."""
    start_time = time.time()
    # Predict using probability and a simple threshold (as per the original logic)
    y_scores = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    predictions = [1 if y > threshold else 0 for y in y_scores]
    end_time = time.time()

    evaluate_model(y_test, predictions, "Clear Text (Non-FHE)")
    print(f"Inference Time (Clear Text): {(end_time - start_time):.4f} seconds")

    # Save clear text predictions to CSV
    df_clear = pd.DataFrame({
        "y_true": y_test,
        "y_pred_clear": predictions
    })
    df_clear.to_csv('predictions_clear_text_dt.csv', index=False)
    print("✅ Clear text predictions saved to 'predictions_clear_text_dt.csv'")
    return df_clear


def run_fhe_evaluation(X_test, y_test, max_depth_limit=10, bits_limit=15):
    """
    Runs a grid search evaluation for the Concrete ML Decision Tree (FHE)
    across different quantization bits and tree depths.
    """
    print("\n" + "="*70)
    print("STARTING HOMOMORPHIC ENCRYPTION (FHE) EVALUATION with Concrete ML")
    print("="*70)
    

    results = []
    all_preds_fhe = []

    # Iterate over different depths and quantization bit widths
    for depth in range(2, max_depth_limit + 1):
        for bits in range(2, bits_limit + 1):
            try:
                print(f"\n[FHE] Test: max_depth={depth}, n_bits={bits}")

                # Initialize Concrete ML Decision Tree
                dt_fhe = ConcreteDT(max_depth=depth, n_bits=bits, random_state=42)

                # Training (Note: For Concrete ML, 'fitting' involves quantization)
                # In the original Italian code, X_test and y_test were used for fitting
                # the FHE model, which might be a small dataset proxy for training or
                # simply the required pre-processing step for the ConcreteDT wrapper.
                start_train = time.time()
                dt_fhe.fit(X_test, y_test)
                end_train = time.time()
                train_time = end_train - start_train

                # Compilation phase (generates the FHE circuit)
                start_compile = time.time()
                # Use a subset of the data for compilation/tracing if the full set is too large
                dt_fhe.compile(X_test)
                end_compile = time.time()
                compile_time = end_compile - start_compile

                # FHE Inference (execution in the encrypted domain)
                start_infer = time.time()
                # The 'fhe="execute"' flag runs the model using the FHE circuit
                y_pred_fhe = dt_fhe.predict(X_test, fhe="execute")
                end_infer = time.time()
                infer_time = end_infer - start_infer

                f1_s = evaluate_model(y_test, y_pred_fhe, f"FHE (Depth={depth}, Bits={bits})")

                print(f"Training/Quantization Time: {train_time:.4f}s")
                print(f"FHE Compilation Time: {compile_time:.4f}s")
                print(f"FHE Inference Time: {infer_time:.4f}s")

                # Store metric results
                results.append({
                    'Depth': depth,
                    'Bits': bits,
                    'FHE_F1': f1_s,
                    'Train_Time (s)': train_time,
                    'Compile_Time (s)': compile_time,
                    'Inference_Time (s)': infer_time,
                    'Status': 'OK'
                })

                # Store predictions for this Depth/Bits configuration
                df_pred_fhe = pd.DataFrame({
                    'Depth': [depth]*len(y_test),
                    'Bits': [bits]*len(y_test),
                    'y_true': y_test,
                    'y_pred_fhe': y_pred_fhe
                })
                all_preds_fhe.append(df_pred_fhe)

            except Exception as e:
                # Handle FHE-specific errors (e.g., depth or bit limit exceeded)
                error_message = f'Error: {str(e)}'
                print(f"FHE ERROR for Depth={depth}, Bits={bits}: {error_message}")
                results.append({
                    'Depth': depth,
                    'Bits': bits,
                    'FHE_F1': 0.0,
                    'Train_Time (s)': 0.0,
                    'Compile_Time (s)': 0.0,
                    'Inference_Time (s)': 0.0,
                    'Status': error_message[:100] # Truncate error message for CSV
                })

    # Save summary of metric results
    df_results = pd.DataFrame(results)
    df_results.to_csv('fhe_evaluation_summary_decision_tree_dt.csv', index=False)
    print("\n✅ FHE evaluation summary saved to 'fhe_evaluation_summary_decision_tree_dt.csv'")

    # Save all FHE predictions
    if all_preds_fhe:
        df_all_preds_fhe = pd.concat(all_preds_fhe, ignore_index=True)
        df_all_preds_fhe.to_csv('predictions_fhe_dt.csv', index=False)
        print("✅ All FHE predictions saved to 'predictions_fhe_dt.csv'")


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # 1. Load, merge, clean, and split the data
    X_train, X_test, y_train, y_test = load_and_merge_data()
    print(f"\nCombined dataset split: {len(X_train)} training samples, {len(X_test)} test samples.")

    # 2. Train the standard (clear text) model
    model_clear = train_and_save_dt(X_train, y_train)

    # 3. Evaluate the clear text model (non-FHE baseline)
    predict_clear_text(model_clear, X_test, y_test)

    # 4. Run FHE evaluation across parameter space
    # Max Depth controls complexity (and thus computation in FHE)
    # Bits controls quantization precision (and thus accuracy)
    run_fhe_evaluation(
        X_test,
        y_test,
        max_depth_limit=10,
        bits_limit=15
    )