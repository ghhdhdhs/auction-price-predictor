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

## 👤 Author

Built by Patrick Mao.
