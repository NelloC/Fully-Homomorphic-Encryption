# Analysis and Prediction of Time Series with Fully Homomorphic Encryption

This repository hosts Machine Learning pipelines designed for the prediction of critical clinical events (Acute Kidney Injury - AKI, Intraoperative Hypotension - IOH, and BIS-MAP composite) using data derived from **VITALDB**. The primary goal is to explore and implement these predictive models using **Concrete ML** to enable inference in the Fully Homomorphic Encryption (FHE) domain, ensuring data privacy and security.

---

## Project Overview

The project is structured into three main domains, each focusing on a different prediction problem and leveraging feature engineering techniques adapted from established research.

### Repository Structure

* .
    * **AKI/**
        * `decision_tree.py` (Decision Tree + FHE for AKI prediction)
        * `logistic.py` (Logistic Regression + FHE for AKI prediction)
    * **BIS_MAP/**
        * `LR_EN_SGD_DT.py` (Linear/Decision Tree models for BIS-MAP prediction)
    * **IOH/**
        * `xgb_IOH.py` (XGBoost + Optuna + FHE pipeline for IOH prediction)
    * `requirements.txt` (Project dependencies)
    * `README.md`

##  Installation and Setup

To run the pipelines locally, you must install the required Python libraries.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/NelloC/Fully-Homomorphic-Encryption.git](https://github.com/NelloC/Fully-Homomorphic-Encryption.git)
    cd Fully-Homomorphic-Encryption
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # on Linux/macOS
    # venv\Scripts\activate   # on Windows
    ```

3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

---

##  Predictive Pipelines and Original Work Credits

The data preprocessing, feature extraction, and initial feature engineering methodologies used in these scripts are **based on and derived from the work presented in the following research repositories**. All credit for the underlying clinical methodology and data handling techniques from VITALDB is attributed to the original authors.

### 1. AKI (Acute Kidney Injury) Prediction

* **Models:** Logistic Regression, Decision Tree
* **Pipeline Files:** `AKI/Classification.py`
* **Pre-processing/Feature Engineering Credit:** The data preparation methodology is based on the work from:
    * **Original Project:** [LFPAP](https://github.com/warriorod/LFPAP)

### 2. BIS-MAP Prediction

* **Models:** Linear models (LR, ElasticNet, SGD) and Decision Tree
* **Pipeline File:** `BIS_MAP/Regression.py`
* **Pre-processing/Feature Engineering Credit:** The methodology for feature extraction and analysis is based on the work from:
    * **Original Project:** [BIS-MAP-Pred](https://github.com/BobAubouin/BIS-MAP-Pred)

### 3. IOH (Intraoperative Hypotension) Prediction

* **Model:** XGBoost Classifier with FHE implementation
* **Pipeline File:** `IOH/Classification.py`
* **Pre-processing/Feature Engineering Credit:** The feature engineering and data preparation methods are based on the work from:
    * **Original Project:** [hypotension\_pred](https://github.com/BobAubouin/hypotension_pred)

---

##  Running the Pipelines

To run a specific pipeline, ensure that all necessary input data files (e.g., CSVs, Parquet files) are placed in the correct locations as specified within the script constants.

**Example Execution (IOH Pipeline):**

```bash
python IOH/Classification.py
