# Auction Price Predictor (Django + Machine Learning)

A Django web application that predicts Japanese vehicle auction prices using
machine-learning models (Linear Regression, Log-Mileage Regression, and XGBoost).  
Includes a real-time prediction API, interactive dashboard, and historical data visualization.

---

## 🚗 Features

- **Machine Learning Models**
  - Linear regression
  - Log-mileage regression 
  - XGBoost 

- **Price Prediction API**
  - Returns predicted price in JPY
  - Includes 90% confidence interval (based on residual quantiles)
  - Supports custom model selection (`linear`, `logm`, `xgb`)

- **Dashboard**
  - Scatter plot of historical auction data  
  - HTMX-powered filtering  
  - Prediction form with live model output

- **Data Pipeline**
  - Cleans auction grades (3.5, 4.0, R, RA)
  - Handles model variants (E400, E43, E53_PRE, E53_FL)
  - Normalizes colors
  - Parses year robustly (2019, "2019(H31)", "2020/01", etc.)

---

## 📦 Installation

### 1. Clone the repo
```bash
git clone https://github.com/ghhdhdhs/auction-price-predictor.git
cd auction-price-predictor

2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run migrations
python manage.py migrate

5. Start the development server
python manage.py runserver

App runs at:
👉 http://127.0.0.1:8000/

👤 Author
Built by Patrick Mao.























