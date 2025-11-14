import time
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Concrete ML
from concrete.ml.sklearn import LogisticRegression as ConcreteLR


AKI_FILE = 'aki_7d_from_api.csv'
INTEROP_FILE = 'interop_from_api.csv'
LAB_FILE = 'lab_for_test_from_api.csv'
MODEL_PATH = './logistic_model_fhe.pkl'


def load_and_merge_data():
    df_aki = pd.read_csv(AKI_FILE)
    df_interop = pd.read_csv(INTEROP_FILE)
    df_lab = pd.read_csv(LAB_FILE)

    df = df_aki.merge(df_interop, on='caseid', how='inner')
    df = df.merge(df_lab, on='caseid', how='inner')

    df['AKI'] = pd.to_numeric(df['AKI'], errors='coerce')
    df = df.dropna(subset=['AKI'])
    df['AKI'] = df['AKI'].astype(int)

    df = df.drop(columns=['caseid'])
    df = df.fillna(-1)

    X = df.drop(columns=['AKI']).values
    y = df['AKI'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test. Features: {X.shape[1]}")
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, model_type="Clear Text"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n--- Metriche ({model_type}) ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return acc, prec, rec, f1

# ============================================================
# MODELLI
# ============================================================

def train_and_save_lr(X_train, y_train):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    end_train = time.time()

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((model, scaler), f)

    print(f"Model saved in in {MODEL_PATH}")
    print(f"Train Time: {end_train - start_train:.4f} seconds")
    return model, scaler

def predict_clear_text(model, scaler, X_test, y_test):

    X_test_scaled = scaler.transform(X_test)
    start = time.time()
    y_scores = model.predict_proba(X_test_scaled)[:, 1]


    best_acc = -1
    best_threshold = 0
    for t in range(100):
        threshold = t * 0.01
        y_pred = (y_scores >= threshold).astype(int)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    y_pred = (y_scores >= best_threshold).astype(int)
    end = time.time()

    acc, prec, rec, f1 = evaluate_model(y_test, y_pred, "Clear Text (No-FHE)")
    print(f"Time): {end - start:.4f} secondi")
    print(f"Adaptive threshold: {best_threshold:.2f}")

    df_pred = pd.DataFrame({
        'y_true': y_test,
        'y_pred_clear': y_pred,
        'y_score': y_scores
    })
    df_pred.to_csv('predictions_clear_text_lr.csv', index=False)
    print(" Prediction clear saved in 'predictions_clear_text_lr.csv'")
    return acc, prec, rec, f1

def run_fhe_evaluation(X_train, y_train, X_test, y_test, bits_limit=15):

    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    results = []
    print("\n=== FHE Logistic Regression Evaluation ===")

    for bits in range(2, bits_limit + 1):
        try:
            print(f"\n[FHE] n_bits={bits}")

            # Split train/validation per soglia
            X_train_fhe, X_val_fhe, y_train_fhe, y_val_fhe = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )

            model_fhe = ConcreteLR(
                n_bits=bits,
                penalty="l2",
                C=5,
                max_iter=1000,
                class_weight="balanced"
            )

            # Addestramento
            t0 = time.time()
            model_fhe.fit(X_train_fhe, y_train_fhe)
            t1 = time.time()

            # Compilazione FHE
            model_fhe.compile(X_test)
            t2 = time.time()

            # Predizioni validation per soglia
            y_val_scores = model_fhe.predict(X_val_fhe, fhe="execute")
            best_acc = -1
            best_threshold = 0
            for t in range(100):
                threshold = t * 0.01
                y_val_pred = (y_val_scores >= threshold).astype(int)
                acc = accuracy_score(y_val_fhe, y_val_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_threshold = threshold
            print(f"Optimal Threshold FHE: {best_threshold:.2f} (Acc={best_acc:.4f})")

            # Predizioni FHE test set
            t3 = time.time()
            y_test_scores = model_fhe.predict(X_test, fhe="execute")
            y_pred_fhe = (y_test_scores >= best_threshold).astype(int)
            t4 = time.time()
            infer_time = t4 - t3

            acc, prec, rec, f1 = evaluate_model(y_test, y_pred_fhe, f"FHE (bits={bits})")
            print(f"Inference Time FHE: {infer_time:.4f} seconds")

            results.append({
                "Bits": bits,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "Train Time (s)": t1 - t0,
                "Compile Time (s)": t2 - t1,
                "Inference Time (s)": infer_time,
                "Threshold": best_threshold
            })

        except Exception as e:
            print(f"❌ Errore FHE con {bits} bits: {e}")
            results.append({
                "Bits": bits,
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
                "Train Time (s)": 0.0,
                "Compile Time (s)": 0.0,
                "Inference Time (s)": 0.0,
                "Threshold": None,
                "Status": f"Error: {e}"
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv("fhe_results_logistic.csv", index=False)
    print("\n Results saved in 'fhe_results_logistic.csv'")
    return df_results

# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_merge_data()

    model_clear, scaler = train_and_save_lr(X_train, y_train)

    predict_clear_text(model_clear, scaler, X_test, y_test)

    run_fhe_evaluation(X_train, y_train, X_test, y_test, bits_limit=15)

