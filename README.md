# Auction Price Predictor (Django + Machine Learning)

A Django web application that predicts Japanese vehicle auction prices using
machine-learning models (Linear Regression, Log-Mileage Regression, and XGBoost).
Includes a real-time prediction API, interactive dashboard, and historical data visualization.

---

## 🚗 Features

- Machine Learning Models:
  - Linear regression
  - Log-mileage regression
  - XGBoost

- Price Prediction API:
  - Returns predicted price in JPY
  - Includes a 90% confidence interval (based on residual quantiles)
  - Supports custom model selection: `linear`, `logm`, `xgb`

- Dashboard:
  - Scatter plot of historical auction data
  - HTMX-powered filtering
  - Prediction form with model output

- Data Pipeline:
  - Cleans auction grades (3.5, 4.0, 4.5, 5.0, R, etc.)
  - Handles model variants (E400, E43, E53_PRE, E53_FL)
  - Normalizes colors
  - Parses year robustly

---

## 📦 Installation

Follow these steps to get the app running on a new machine.

### 1. Clone the repo

    git clone https://github.com/ghhdhdhs/auction-price-predictor.git
    cd auction-price-predictor

### 2. Create and activate a virtual environment (Windows)

    python -m venv .venv
    .venv\Scripts\activate

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Run migrations

    python manage.py migrate

(Optional, if you want admin access)

    python manage.py createsuperuser

### 5. Import auction data from cars.csv

Make sure `cars.csv` is in the project root (the same folder as `manage.py`), then run:

    python scripts/import_cars_csv.py

You should see a summary similar to:

    ✅ Import complete.
       Chassis created: X
       CarSale created: Y, updated: Z
       Skipped (unknown model): ...
       Skipped (missing/zero final_price): ...

This populates Maker, Chassis, and CarSale tables from your CSV.

### 6. Train all prediction models

Open the Django shell:

    python manage.py shell

Then run the following Python code inside the shell:

    from pricepred.ml_basic import train_basic, train_logm, train_xgb

    print(train_basic())    # Linear regression (mileage)
    print(train_logm())     # Linear regression (log-mileage) — default model

    try:
        print(train_xgb())  # Optional XGBoost model
    except RuntimeError as e:
        print("XGBoost not available:", e)

    exit()

After this, model files (.pkl) will appear in the `models/` directory:
- models/basic_model.pkl
- models/logm_model.pkl
- models/xgb_model.pkl 

### 7. Start the development server

    python manage.py runserver

The app runs at:

- Dashboard → http://127.0.0.1:8000/
- Admin (if you created a superuser) → http://127.0.0.1:8000/admin/

---


## 📊 Model Evaluation (Optional)

You can evaluate and compare model performance (Linear, Log-Mileage, XGBoost)
using the built-in cross-validation functions in `pricepred.ml_basic`.

These functions measure:
- MAE   = Mean Absolute Error  
- RMSE  = Root Mean Squared Error  
- R²    = Coefficient of Determination  
- MAPE  = Mean Absolute Percentage Error  

Evaluation does NOT require trained .pkl model files — it trains temporary models internally.

---

### 🔍 1. Leave-One-Out Cross-Validation (LOOCV)

LOOCV is the most accurate form of cross-validation.  
Each row in the dataset becomes its own test set.

To run LOOCV:

1. Open Django shell:

       python manage.py shell

2. Run:

       from pricepred.ml_basic import evaluate_models_all, print_eval_table

       results = evaluate_models_all(cv="loocv")
       print_eval_table(results, decimals=4)

Example output shape:

       Model        | MAE     | RMSE    | R2    | MAPE
       ------------ | ------- | ------- | ----- | -------
       linear       | ...     | ...     | ...   | ...
       log_mileage  | ...     | ...     | ...   | ...
       xgboost      | ...     | ...     | ...   | ...


---

### 🔍 2. K-Fold Cross-Validation (default k=5)

K-Fold CV splits the data into *k* equal parts (folds).  
This is faster than LOOCV and still gives stable results.

To run 5-fold CV:

1. Open Django shell:

       python manage.py shell

2. Run:

       from pricepred.ml_basic import evaluate_models_all, print_eval_table

       results = evaluate_models_all(cv="kfold", k=5, random_state=42)
       print_eval_table(results, decimals=4)
You can change **k** to any number, e.g. k=10:


---

This evaluation step is mainly for developers and data enthusiasts who want to understand which model performs best on the current dataset.


## ⚠️ Disclaimer
This project is for personal and educational use only.
The prediction results are approximate and may be inaccurate, and should not be used as financial advice or relied upon for real-world purchasing decisions.


## 👤 Author

Built by Patrick Mao. <br>
Debugged by future Patrick Mao.
