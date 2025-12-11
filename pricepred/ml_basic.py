from __future__ import annotations

"""
Core machine-learning utilities for auction price prediction.

This module provides a complete pipeline for:
    • Fetching and normalising historical auction data
    • Feature engineering and one-hot encoding
    • Training multiple regression models:
          - Linear regression (raw mileage)
          - Linear regression with log(mileage)
          - XGBoost regressor (optional dependency)
    • Estimating prediction intervals using empirical residual quantiles
    • Running model inference with consistent preprocessing
    • Cross-validated evaluation (MAE / RMSE / R² / MAPE)

------------------------------------------------------------------------------
Key Design Principles
------------------------------------------------------------------------------
1. A single source of truth for feature extraction
   ------------------------------------------------
   _fetch_training_df() produces the canonical cleaned DataFrame with:
        ['mileage', 'ovrl_grade', 'Model', 'Color', 'final_price']
   All models and UI components consume this same schema to guarantee
   consistency across training, prediction, and dashboard visualisation.

2. Encapsulated preprocessing
   ------------------------------------------------
   Functions like:
       - _normalize_payload()
       - _normalize_color_for_payload()
       - _row_to_X()
   ensure that model inputs are transformed exactly as during training.
   This prevents subtle mismatches between training and inference.

3. Residual-based confidence intervals
   ------------------------------------------------
   Each trained model stores quantile statistics of residuals:
        - global
        - grouped by model
        - grouped by model+grade
   predict_*_with_ci() returns both point estimates and an approximate
   90% prediction interval using these stored quantiles.

4. Model-agnostic API
   ------------------------------------------------
   Training functions all return a bundle with:
        'model', 'columns', 'grade_levels', 'model_levels', 'color_levels'
   Prediction functions accept raw payloads and return clean numerical
   outputs ready for JSON serialization.

------------------------------------------------------------------------------
Intended Usage
------------------------------------------------------------------------------
• Called by Django views (api_predict) for real-time inference.
• Called manually via CLI (__main__) for local debugging.
• Generates model bundles (.pkl) stored in /models.

------------------------------------------------------------------------------
Note
------------------------------------------------------------------------------
This module intentionally avoids any Django imports (except the CarSale ORM
model in _fetch_training_df). All ML logic remains framework-agnostic so that
the prediction system can be reused in notebooks, scripts, or API servers.

"""


from pathlib import Path
from typing import Dict, Any, List, Optional
import re  # robust year parsing from mixed formats

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Optional: xgboost (gracefully handled if not installed)
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# If this lives inside a Django app package, keep the relative import:
from .models import CarSale


# ------------------------------
# Paths
# ------------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH_LIN = MODEL_DIR / "basic_model.pkl"        # Linear regression (raw mileage)
MODEL_PATH_XGB = MODEL_DIR / "xgb_model.pkl"          # XGBoost
MODEL_PATH_LOGM = MODEL_DIR / "logm_model.pkl"        # Linear regression (log-mileage)


# ------------------------------
# Helpers
# ------------------------------
def _parse_year(year_raw: Any) -> Optional[int]:
    """
    Parse a 4-digit year from various raw formats.

    Handles examples such as:
      - 2019, 2019.0
      - "2019", "2019.0"
      - "2019(H31)", "2024/01", etc.

    Returns:
        int year if a 4-digit year can be extracted, otherwise None.
    """
    if year_raw is None or (isinstance(year_raw, float) and pd.isna(year_raw)):
        return None
    try:
        # Covers simple int/float inputs like 2019 or "2019"
        return int(year_raw)
    except Exception:
        pass

    # Fallback: regex search for any 4 consecutive digits
    m = re.search(r"\d{4}", str(year_raw))
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def _normalize_color_from_db(series: pd.Series) -> pd.Series:
    """
    Normalise the CarSale 'color' field into a small, stable set of labels.

    The DB may contain values like "White", "Black", "White (Pearl)", etc.
    This function maps them into canonical labels:
        White, Black, Silver, Blue, Gray, Red, Other

    Any unknown or null values fall back to "Other".
    """
    if series.isnull().all():
        return pd.Series(["Other"] * len(series), index=series.index)

    s = series.fillna("Other").astype(str).str.strip()

    def _map_one(v: str) -> str:
        up = v.upper()
        if "WHITE" in up:
            return "White"
        if "BLACK" in up:
            return "Black"
        if "SILVER" in up:
            return "Silver"
        if "BLUE" in up or "NAVY" in up:
            return "Blue"
        if "GRAY" in up or "GREY" in up:
            return "Gray"
        if "RED" in up:
            return "Red"
        if "OTHER" in up:
            return "Other"
        return "Other"

    return s.map(_map_one)


# ------------------------------
# Data fetching & normalization
# ------------------------------
def _fetch_training_df() -> pd.DataFrame:
    """
    Fetch and preprocess training data from the CarSale model.

    Returns a DataFrame with columns:
        - mileage      (float)
        - ovrl_grade   (str; normalised auction grade, e.g. "4.0", "4.5", "R")
        - Model        (str; 'E400', 'E43', 'E53_PRE', 'E53_FL')
        - Color        (str; 'White', 'Black', 'Silver', 'Blue', 'Gray', 'Red', 'Other')
        - final_price  (float; JPY)

    Rows with missing or unusable values are dropped.
    """

    qs = (
        CarSale.objects
        .select_related("chassis_code")
        .values(
            "mileage",
            "auction_grade",
            "final_price",
            "chassis_code__variant",
            "year",
            "color",
        )
    )
    df = pd.DataFrame(list(qs))
    if df.empty:
        raise ValueError("No rows in CarSale.")

    # Basic numeric conversions
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
    df["final_price"] = pd.to_numeric(df["final_price"], errors="coerce")
    # 'year' remains raw; we parse it with _parse_year only where required.

    def _norm_grade(v) -> Optional[str]:
        """
        Normalise auction_grade into a string category.

        Behaviour:
            - Numeric-like: "4.5" -> "4.5", "4" -> "4.0"
            - Letter-like:  keep as uppercase string ("R", "RA", etc.)
            - Missing/empty: None
        """
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None

        s = str(v).strip().upper()
        if not s:
            return None

        # Try numeric first (e.g. "3.5", "4")
        try:
            return f"{float(s):.1f}"
        except Exception:
            # Non-numeric: treat as categorical grade label, e.g. "R", "RA"
            return s

    df["ovrl_grade"] = df["auction_grade"].map(_norm_grade)

    def _model_from_row(row) -> Optional[str]:
        """
        Infer the internal Model label from chassis variant and year.

        Rules:
            - E53_PRE : E53 with year < 2021 or unknown year
            - E53_FL  : E53 with year >= 2021
            - E43, E400: matched from chassis variant string

        Returns:
            One of {"E400", "E43", "E53_PRE", "E53_FL"} or None if unknown.
        """
        s = (row.get("chassis_code__variant") or "").upper()
        year_int = _parse_year(row.get("year"))

        # E53 handling (split by pre / facelift)
        if "E53" in s:
            if year_int is not None and year_int >= 2021:
                return "E53_FL"
            else:
                return "E53_PRE"

        # Other models
        if "E43" in s:
            return "E43"
        if "E400" in s:
            return "E400"

        return None

    df["Model"] = df.apply(_model_from_row, axis=1)

    # Normalised Color from DB
    df["Color"] = _normalize_color_from_db(df["color"])

    # Drop unusable rows
    df = df.dropna(subset=["mileage", "final_price", "ovrl_grade", "Model", "Color"])
    if df.empty:
        raise ValueError("No usable rows after filtering for mileage/grade/Model/Color.")

    return df[["mileage", "ovrl_grade", "Model", "Color", "final_price"]]


def _freeze_and_dummify(df: pd.DataFrame):
    """
    One-hot encode the main features and freeze categorical levels.

    Args:
        df: DataFrame with columns ["mileage", "ovrl_grade", "Model", "Color", "final_price"].

    Returns:
        X              : Design matrix (np.ndarray-like) with dummy variables.
        y              : Target array of final prices.
        grade_levels   : Sorted list of grade categories.
        model_levels   : Sorted list of model categories.
        color_levels   : Sorted list of colour categories.
    """
    grade_levels: List[str] = sorted(df["ovrl_grade"].unique().tolist())
    model_levels: List[str] = sorted(df["Model"].unique().tolist())
    color_levels: List[str] = sorted(df["Color"].unique().tolist())

    X_raw = df[["mileage", "ovrl_grade", "Model", "Color"]].copy()
    X_raw["ovrl_grade"] = pd.Categorical(X_raw["ovrl_grade"], categories=grade_levels, ordered=False)
    X_raw["Model"]      = pd.Categorical(X_raw["Model"],      categories=model_levels, ordered=False)
    X_raw["Color"]      = pd.Categorical(X_raw["Color"],      categories=color_levels, ordered=False)

    X = pd.get_dummies(X_raw, drop_first=True)
    y = df["final_price"].values.astype(float)
    return X, y, grade_levels, model_levels, color_levels


def _normalize_grade_for_payload(raw_grade: Any) -> str:
    """
    Normalise ovrl_grade from user payload.

    Grade can be numeric (e.g. "4.5", 4.5) or a letter code ("R", "RA").
    This mirrors the logic used during training in _norm_grade().

    Raises:
        ValueError if the grade is missing or empty.
    """
    if raw_grade is None:
        raise ValueError("ovrl_grade is required")

    s = str(raw_grade).strip().upper()
    if not s:
        raise ValueError("ovrl_grade is empty")

    try:
        return f"{float(s):.1f}"
    except Exception:
        return s  # e.g. "R", "RA"


def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise a single prediction payload to match the training schema.

    Expected payload keys:
        - "Model"      : internal label, one of {"E400", "E43", "E53_PRE", "E53_FL"}
        - "mileage"    : numeric or numeric-like string
        - "ovrl_grade" : grade string or number (see _normalize_grade_for_payload)

    Colour is handled separately by _normalize_color_for_payload.

    Returns:
        A dict with normalised:
            - mileage   (float)
            - ovrl_grade (str)
            - Model      (str, uppercased)
    """
    # mileage
    try:
        mileage = float(payload["mileage"])
    except Exception as e:
        raise ValueError(f"Invalid mileage in payload: {payload.get('mileage')}") from e

    grade_str = _normalize_grade_for_payload(payload.get("ovrl_grade"))

    model_str = str(payload["Model"]).upper().strip()

    return {
        "mileage": mileage,
        "ovrl_grade": grade_str,
        "Model": model_str,
    }


def _normalize_color_for_payload(
    payload: Dict[str, Any],
    color_levels: List[str],
    default_color: Optional[str]
) -> str:
    """
    Decide which Color category to use for prediction.

    Behaviour:
        - If the user provides a specific colour (e.g. "White", "Black"),
          map it to the closest known category.
        - If the user leaves colour empty or uses an "any" style value
          ("Any", "All colours", etc.), use default_color (usually the
          most common colour from training).
        - If the mapped value is not in the training levels, fall back to
          "Other" if available, else default_color, else the first level.

    The payload may contain either "colour" or "color".
    """
    raw = payload.get("colour") or payload.get("color")

    # Interpret "any / all" or missing as default colour
    if raw is None:
        use_any = True
    else:
        txt = str(raw).strip().upper()
        use_any = (
            txt == ""
            or txt in {
                "ALL", "ANY", "ANY COLOUR", "ANY COLOR",
                "ALL COLOURS", "ALL COLORS"
            }
        )

    if use_any:
        if default_color is not None:
            return default_color
        # Fallback: some colour must be chosen
        return color_levels[0] if color_levels else "Other"

    # User gave a specific colour
    txt = str(raw).strip().upper()

    # Rough normalisation similar to the DB helper
    if "WHITE" in txt:
        cand = "White"
    elif "BLACK" in txt:
        cand = "Black"
    elif "SILVER" in txt:
        cand = "Silver"
    elif "BLUE" in txt or "NAVY" in txt:
        cand = "Blue"
    elif "GRAY" in txt or "GREY" in txt:
        cand = "Gray"
    elif "RED" in txt:
        cand = "Red"
    else:
        cand = "Other"

    # If this normalised label was never seen in training, back off
    if cand not in color_levels:
        if "Other" in color_levels:
            return "Other"
        if default_color is not None:
            return default_color
        return color_levels[0] if color_levels else "Other"

    return cand


def _row_to_X(
    norm: Dict[str, Any],
    cols: List[str],
    grades: List[str],
    models: List[str],
    colors: List[str],
) -> pd.DataFrame:
    """
    Build a one-row design matrix aligned with the training feature columns.

    Args:
        norm   : Normalised payload dict with keys ["mileage", "ovrl_grade", "Model", "Color"].
        cols   : List of feature names used during training.
        grades : Frozen grade_levels from training.
        models : Frozen model_levels from training.
        colors : Frozen color_levels from training.

    Returns:
        A one-row DataFrame with dummy variables, ordered exactly as in `cols`.
    """
    row = pd.DataFrame([norm])
    row["ovrl_grade"] = pd.Categorical(row["ovrl_grade"], categories=grades, ordered=False)
    row["Model"]      = pd.Categorical(row["Model"],      categories=models, ordered=False)
    row["Color"]      = pd.Categorical(row["Color"],      categories=colors, ordered=False)
    X = pd.get_dummies(row, drop_first=True).reindex(columns=cols, fill_value=0.0)
    return X


# ------------------------------
# Residual → CI helper (shared)
# ------------------------------
def _compute_ci_from_residuals(
    norm: Dict[str, Any],
    bundle: Dict[str, Any],
    y_hat: float,
    fallback_band_jpy: float = 50000.0,
    fallback_pct: float = 0.15,
) -> tuple[float, float]:
    """
    Compute an approximate 90% prediction interval using stored residuals.

    The model bundle may contain residual metadata under `bundle["residuals"]`:
        - "by_model_grade": dict[(Model||Grade) -> (q05, q95)]
        - "by_model"      : dict[Model -> (q05, q95)]
        - "global_q05"    : float (5th percentile of residuals)
        - "global_q95"    : float (95th percentile of residuals)

    Strategy:
        1. First try a Model+Grade-specific band.
        2. Fall back to model-only band.
        3. Fall back to global band.
        4. If no metadata is present (older bundles), use a symmetric band:
           max(±fallback_pct * |y_hat|, ±fallback_band_jpy).

    Returns:
        (low, high) interval bounds, clipped at 0 JPY.
    """
    resid_meta = bundle.get("residuals")
    if not isinstance(resid_meta, dict):
        band = max(fallback_pct * abs(y_hat), fallback_band_jpy)
        low = max(y_hat - band, 0.0)
        high = max(y_hat + band, 0.0)
        return low, high

    by_mg = resid_meta.get("by_model_grade") or {}
    by_m  = resid_meta.get("by_model") or {}
    g_q05 = resid_meta.get("global_q05")
    g_q95 = resid_meta.get("global_q95")

    mg_key = f"{norm['Model']}||{norm['ovrl_grade']}"
    if mg_key in by_mg:
        q05, q95 = by_mg[mg_key]
    elif norm["Model"] in by_m:
        q05, q95 = by_m[norm["Model"]]
    elif g_q05 is not None and g_q95 is not None:
        q05, q95 = g_q05, g_q95
    else:
        # No usable metadata, revert to simple symmetric band
        band = max(fallback_pct * abs(y_hat), fallback_band_jpy)
        low = max(y_hat - band, 0.0)
        high = max(y_hat + band, 0.0)
        return low, high

    low = y_hat + float(q05)
    high = y_hat + float(q95)
    low = max(low, 0.0)
    high = max(high, 0.0)
    return low, high


# ------------------------------
# Linear Regression (basic)
# ------------------------------
def train_basic(min_rows: int = 6) -> Dict[str, Any]:
    """
    Train the baseline linear regression model on raw mileage.

    Model:
        final_price ~ mileage + ovrl_grade + Model + Color

    Also computes empirical residual quantiles, which are stored in the
    bundle and later used to derive approximate 90% prediction intervals.

    Args:
        min_rows: Minimum number of usable training rows required.

    Returns:
        A summary dict with basic training metadata.
    """
    df = _fetch_training_df()
    if len(df) < min_rows:
        raise ValueError(f"Need ≥{min_rows} rows, have {len(df)}.")

    X, y, grade_levels, model_levels, color_levels = _freeze_and_dummify(df)

    # Most common colour for "Any colour" default
    default_color = df["Color"].value_counts().idxmax()

    model = LinearRegression().fit(X, y)

    # ---------- residuals + quantiles for 90% interval ----------
    y_hat = model.predict(X)
    df["_resid"] = y - y_hat

    # Global residual quantiles
    global_q05, global_q95 = np.quantile(df["_resid"], [0.05, 0.95])

    # By (Model, Grade)
    resid_by_mg: Dict[str, tuple[float, float]] = {}
    for (m, g), grp in df.groupby(["Model", "ovrl_grade"]):
        if len(grp) < max(8, min_rows):
            continue
        q05, q95 = np.quantile(grp["_resid"], [0.05, 0.95])
        key = f"{m}||{g}"
        resid_by_mg[key] = (float(q05), float(q95))

    # By Model only
    resid_by_m: Dict[str, tuple[float, float]] = {}
    for m, grp in df.groupby("Model"):
        if len(grp) < max(8, min_rows):
            continue
        q05, q95 = np.quantile(grp["_resid"], [0.05, 0.95])
        resid_by_m[str(m)] = (float(q05), float(q95))

    residual_meta = {
        "global_q05": float(global_q05),
        "global_q95": float(global_q95),
        "by_model_grade": resid_by_mg,
        "by_model": resid_by_m,
    }

    bundle = {
        "model": model,
        "columns": X.columns.tolist(),
        "grade_levels": grade_levels,
        "model_levels": model_levels,
        "color_levels": color_levels,
        "meta": {
            "n_rows": int(len(df)),
            "baseline": {
                "ovrl_grade": grade_levels[0] if grade_levels else None,
                "Model": model_levels[0] if model_levels else None,
            },
            "default_color": default_color,
        },
        "residuals": residual_meta,
    }
    joblib.dump(bundle, MODEL_PATH_LIN)
    return {
        "trained_on": int(len(df)),
        "features": int(X.shape[1]),
        "grade_levels": grade_levels,
        "model_levels": model_levels,
        "color_levels": color_levels,
        "default_color": default_color,
        "model_path": str(MODEL_PATH_LIN),
        "has_residuals": True,
    }


def predict_basic_with_ci(payload: Dict[str, Any]) -> tuple[float, float, float]:
    """
    Predict using the basic linear model and return a 90% prediction interval.

    Args:
        payload: Dict-like input with keys:
            - "Model"
            - "mileage"
            - "ovrl_grade"
            - "colour" / "color" (optional; defaults to most common colour)

    Returns:
        (y_hat, low_90, high_90) in JPY.
    """
    if not MODEL_PATH_LIN.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH_LIN}. Train the model first.")
    bundle = joblib.load(MODEL_PATH_LIN)
    norm = _normalize_payload(payload)

    grades: List[str] = bundle["grade_levels"]
    models: List[str] = bundle["model_levels"]
    colors: List[str] = bundle.get("color_levels", [])
    cols:   List[str] = bundle["columns"]
    meta:   Dict[str, Any] = bundle.get("meta", {})
    default_color: Optional[str] = meta.get("default_color")

    # Decide which Color to use (respecting "Any colour")
    color_str = _normalize_color_for_payload(payload, colors, default_color)
    norm["Color"] = color_str

    if norm["ovrl_grade"] not in grades:
        raise ValueError(f"ovrl_grade '{norm['ovrl_grade']}' was not seen in training. Seen: {grades}")
    if norm["Model"] not in models:
        raise ValueError(f"Model '{norm['Model']}' was not seen in training. Seen: {models}")
    if norm["Color"] not in colors:
        raise ValueError(f"Color '{norm['Color']}' was not seen in training. Seen: {colors}")

    X = _row_to_X(norm, cols, grades, models, colors)
    model = bundle["model"]
    y_hat = float(model.predict(X)[0])

    low, high = _compute_ci_from_residuals(norm, bundle, y_hat)
    return y_hat, low, high


def predict_basic(payload: Dict[str, Any]) -> float:
    """
    Backwards-compatible helper that returns only the point prediction.

    Use predict_basic_with_ci(...) if you also need interval bounds.
    """
    y_hat, _, _ = predict_basic_with_ci(payload)
    return y_hat


# ------------------------------
# Linear Regression with log-mileage
# ------------------------------
def train_logm(min_rows: int = 6) -> Dict[str, Any]:
    """
    Train a linear regression model using log(mileage).

    Model:
        final_price ~ log(mileage) + ovrl_grade + Model + Color

    As with train_basic, residual quantiles are stored for later use
    in constructing approximate 90% prediction intervals.

    Args:
        min_rows: Minimum number of usable training rows required.

    Returns:
        A summary dict with basic training metadata.
    """
    df = _fetch_training_df()
    if len(df) < min_rows:
        raise ValueError(f"Need ≥{min_rows} rows, have {len(df)}.")

    # Add log_mileage feature
    df["log_mileage"] = np.log1p(df["mileage"])

    # Freeze category levels
    grade_levels: List[str] = sorted(df["ovrl_grade"].unique().tolist())
    model_levels: List[str] = sorted(df["Model"].unique().tolist())
    color_levels: List[str] = sorted(df["Color"].unique().tolist())

    # Most common colour for "Any colour" default
    default_color = df["Color"].value_counts().idxmax()

    # Design matrix
    X_raw = df[["log_mileage", "ovrl_grade", "Model", "Color"]].copy()
    X_raw["ovrl_grade"] = pd.Categorical(X_raw["ovrl_grade"], categories=grade_levels, ordered=False)
    X_raw["Model"]      = pd.Categorical(X_raw["Model"],      categories=model_levels, ordered=False)
    X_raw["Color"]      = pd.Categorical(X_raw["Color"],      categories=color_levels, ordered=False)

    X = pd.get_dummies(X_raw, drop_first=True)
    y = df["final_price"].values.astype(float)

    model = LinearRegression().fit(X, y)

    # ---------- residuals + quantiles ----------
    y_hat = model.predict(X)
    df["_resid"] = y - y_hat

    global_q05, global_q95 = np.quantile(df["_resid"], [0.05, 0.95])

    resid_by_mg: Dict[str, tuple[float, float]] = {}
    for (m, g), grp in df.groupby(["Model", "ovrl_grade"]):
        if len(grp) < max(8, min_rows):
            continue
        q05, q95 = np.quantile(grp["_resid"], [0.05, 0.95])
        key = f"{m}||{g}"
        resid_by_mg[key] = (float(q05), float(q95))

    resid_by_m: Dict[str, tuple[float, float]] = {}
    for m, grp in df.groupby("Model"):
        if len(grp) < max(8, min_rows):
            continue
        q05, q95 = np.quantile(grp["_resid"], [0.05, 0.95])
        resid_by_m[str(m)] = (float(q05), float(q95))

    residual_meta = {
        "global_q05": float(global_q05),
        "global_q95": float(global_q95),
        "by_model_grade": resid_by_mg,
        "by_model": resid_by_m,
    }

    bundle = {
        "model": model,
        "columns": X.columns.tolist(),
        "grade_levels": grade_levels,
        "model_levels": model_levels,
        "color_levels": color_levels,
        "meta": {
            "n_rows": int(len(df)),
            "baseline": {
                "ovrl_grade": grade_levels[0] if grade_levels else None,
                "Model": model_levels[0] if model_levels else None,
            },
            "transform": "log_mileage",
            "default_color": default_color,
        },
        "residuals": residual_meta,
    }
    joblib.dump(bundle, MODEL_PATH_LOGM)
    return {
        "trained_on": int(len(df)),
        "features": int(X.shape[1]),
        "grade_levels": grade_levels,
        "model_levels": model_levels,
        "color_levels": color_levels,
        "default_color": default_color,
        "model_path": str(MODEL_PATH_LOGM),
        "has_residuals": True,
    }


def predict_logm_with_ci(payload: Dict[str, Any]) -> tuple[float, float, float]:
    """
    Predict final_price using the log-mileage model and return interval bounds.

    Args:
        payload: Dict-like input with keys:
            - "Model"
            - "mileage"
            - "ovrl_grade"
            - "colour" / "color" (optional; defaults to most common colour)

    Returns:
        (y_hat, low_90, high_90) in JPY.
    """
    if not MODEL_PATH_LOGM.exists():
        raise FileNotFoundError(f"No log-mileage model at {MODEL_PATH_LOGM}, train it first.")

    bundle = joblib.load(MODEL_PATH_LOGM)
    norm = _normalize_payload(payload)

    grades: List[str] = bundle["grade_levels"]
    models: List[str] = bundle["model_levels"]
    colors: List[str] = bundle.get("color_levels", [])
    cols:   List[str] = bundle["columns"]
    meta:   Dict[str, Any] = bundle.get("meta", {})
    default_color: Optional[str] = meta.get("default_color")

    # Decide Color (respect "Any colour")
    color_str = _normalize_color_for_payload(payload, colors, default_color)
    norm["Color"] = color_str

    if norm["ovrl_grade"] not in grades:
        raise ValueError(f"ovrl_grade '{norm['ovrl_grade']}' was not seen in training. Seen: {grades}")
    if norm["Model"] not in models:
        raise ValueError(f"Model '{norm['Model']}' was not seen in training. Seen: {models}")
    if norm["Color"] not in colors:
        raise ValueError(f"Color '{norm['Color']}' was not seen in training. Seen: {colors}")

    # Build row with log(mileage)
    row = pd.DataFrame([{
        "log_mileage": np.log1p(norm["mileage"]),
        "ovrl_grade": norm["ovrl_grade"],
        "Model": norm["Model"],
        "Color": norm["Color"],
    }])

    row["ovrl_grade"] = pd.Categorical(row["ovrl_grade"], categories=grades)
    row["Model"]      = pd.Categorical(row["Model"],      categories=models)
    row["Color"]      = pd.Categorical(row["Color"],      categories=colors)

    X = pd.get_dummies(row, drop_first=True).reindex(columns=cols, fill_value=0.0)

    model = bundle["model"]
    y_hat = float(model.predict(X)[0])

    low, high = _compute_ci_from_residuals(norm, bundle, y_hat)
    return y_hat, low, high


def predict_logm(payload: Dict[str, Any]) -> float:
    """
    Backwards-compatible helper that returns only the point prediction.

    Use predict_logm_with_ci(...) if you also need interval bounds.
    """
    y_hat, _, _ = predict_logm_with_ci(payload)
    return y_hat


# ------------------------------
# XGBoost
# ------------------------------
def train_xgb(min_rows: int = 6, random_state: int = 42) -> Dict[str, Any]:
    """
    Train an XGBoost regressor on the same feature set as the basic model.

    Args:
        min_rows    : Minimum number of usable training rows required.
        random_state: Seed for reproducibility.

    Returns:
        A summary dict with basic training metadata.

    Raises:
        RuntimeError if xgboost is not installed.
    """
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not installed. Run: pip install xgboost")

    df = _fetch_training_df()
    if len(df) < min_rows:
        raise ValueError(f"Need ≥{min_rows} rows, have {len(df)}.")

    X, y, grade_levels, model_levels, color_levels = _freeze_and_dummify(df)

    default_color = df["Color"].value_counts().idxmax()

    # Conservative defaults; can be tuned when more data is available
    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=0,  # let xgboost pick threads; set >0 if you want control
    )
    model.fit(X, y)

    # ---------- residuals + quantiles ----------
    y_hat = model.predict(X)
    df["_resid"] = y - y_hat

    global_q05, global_q95 = np.quantile(df["_resid"], [0.05, 0.95])

    resid_by_mg: Dict[str, tuple[float, float]] = {}
    for (m, g), grp in df.groupby(["Model", "ovrl_grade"]):
        if len(grp) < max(8, min_rows):
            continue
        q05, q95 = np.quantile(grp["_resid"], [0.05, 0.95])
        key = f"{m}||{g}"
        resid_by_mg[key] = (float(q05), float(q95))

    resid_by_m: Dict[str, tuple[float, float]] = {}
    for m, grp in df.groupby("Model"):
        if len(grp) < max(8, min_rows):
            continue
        q05, q95 = np.quantile(grp["_resid"], [0.05, 0.95])
        resid_by_m[str(m)] = (float(q05), float(q95))

    residual_meta = {
        "global_q05": float(global_q05),
        "global_q95": float(global_q95),
        "by_model_grade": resid_by_mg,
        "by_model": resid_by_m,
    }

    bundle = {
        "model": model,
        "columns": X.columns.tolist(),
        "grade_levels": grade_levels,
        "model_levels": model_levels,
        "color_levels": color_levels,
        "meta": {
            "n_rows": int(len(df)),
            "algorithm": "xgboost",
            "default_color": default_color,
        },
        "residuals": residual_meta,
    }
    joblib.dump(bundle, MODEL_PATH_XGB)
    return {
        "trained_on": int(len(df)),
        "features": int(X.shape[1]),
        "grade_levels": grade_levels,
        "model_levels": model_levels,
        "color_levels": color_levels,
        "default_color": default_color,
        "model_path": str(MODEL_PATH_XGB),
        "has_residuals": True,
    }


def predict_xgb_with_ci(payload: Dict[str, Any]) -> tuple[float, float, float]:
    """
    Predict final_price using the XGBoost model and return interval bounds.

    Args:
        payload: Dict-like input with keys:
            - "Model"
            - "mileage"
            - "ovrl_grade"
            - "colour" / "color" (optional; defaults to most common colour)

    Returns:
        (y_hat, low_90, high_90) in JPY.

    Raises:
        RuntimeError      if xgboost is not installed.
        FileNotFoundError if the XGBoost model file does not exist.
    """
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not installed. Run: pip install xgboost")
    if not MODEL_PATH_XGB.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH_XGB}. Train the model first.")

    bundle = joblib.load(MODEL_PATH_XGB)
    norm = _normalize_payload(payload)

    grades: List[str] = bundle["grade_levels"]
    models: List[str] = bundle["model_levels"]
    colors: List[str] = bundle.get("color_levels", [])
    cols:   List[str] = bundle["columns"]
    meta:   Dict[str, Any] = bundle.get("meta", {})
    default_color: Optional[str] = meta.get("default_color")

    # Decide Color (respect "Any colour")
    color_str = _normalize_color_for_payload(payload, colors, default_color)
    norm["Color"] = color_str

    if norm["ovrl_grade"] not in grades:
        raise ValueError(f"ovrl_grade '{norm['ovrl_grade']}' was not seen in training. Seen: {grades}")
    if norm["Model"] not in models:
        raise ValueError(f"Model '{norm['Model']}' was not seen in training. Seen: {models}")
    if norm["Color"] not in colors:
        raise ValueError(f"Color '{norm['Color']}' was not seen in training. Seen: {colors}")

    X = _row_to_X(norm, cols, grades, models, colors)
    model = bundle["model"]
    y_hat = float(model.predict(X)[0])

    low, high = _compute_ci_from_residuals(norm, bundle, y_hat)
    return y_hat, low, high


def predict_xgb(payload: Dict[str, Any]) -> float:
    """
    Backwards-compatible helper that returns only the point prediction.

    Use predict_xgb_with_ci(...) if you also need interval bounds.
    """
    y_hat, _, _ = predict_xgb_with_ci(payload)
    return y_hat


# =========================
# Model evaluation helpers
# =========================
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_mape(y_true, y_pred, eps: float = 1e-9) -> float:
    """
    Compute a numerically safe MAPE, protecting against division by zero.

    Args:
        y_true: Iterable of true target values.
        y_pred: Iterable of predicted values.
        eps   : Small positive value used as a lower bound in the denominator.

    Returns:
        Mean Absolute Percentage Error as a float.
    """
    y_true = pd.Series(y_true, dtype=float).values
    y_pred = pd.Series(y_pred, dtype=float).values
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def _build_design_from_df(
    df: pd.DataFrame,
    grade_levels: Optional[List[str]] = None,
    model_levels: Optional[List[str]] = None,
    color_levels: Optional[List[str]] = None,
):
    """
    Build a design matrix X and target y from a DataFrame.

    If category levels are provided, they are used to freeze ordering.
    Otherwise, levels are inferred from the DataFrame.

    Returns:
        X, y, grade_levels, model_levels, color_levels
    """
    if grade_levels is None:
        grade_levels = sorted(df["ovrl_grade"].unique().tolist())
    if model_levels is None:
        model_levels = sorted(df["Model"].unique().tolist())
    if color_levels is None:
        color_levels = sorted(df["Color"].unique().tolist())

    X_raw = df[["mileage", "ovrl_grade", "Model", "Color"]].copy()
    X_raw["ovrl_grade"] = pd.Categorical(X_raw["ovrl_grade"], categories=grade_levels, ordered=False)
    X_raw["Model"]      = pd.Categorical(X_raw["Model"],      categories=model_levels, ordered=False)
    X_raw["Color"]      = pd.Categorical(X_raw["Color"],      categories=color_levels, ordered=False)

    X = pd.get_dummies(X_raw, drop_first=True)
    y = df["final_price"].astype(float).values
    return X, y, grade_levels, model_levels, color_levels


def _kfold(df: pd.DataFrame, cv: str = "loocv", k: int = 5, seed: int = 42) -> KFold:
    """
    Construct a KFold-like splitter.

    Args:
        df : DataFrame whose length determines the number of samples.
        cv : "loocv" for leave-one-out; otherwise standard K-fold.
        k  : Number of folds when cv != "loocv".
        seed: Random seed when shuffling in standard K-fold.

    Returns:
        An instance of sklearn.model_selection.KFold.
    """
    return (KFold(n_splits=len(df), shuffle=False)
            if cv.lower() == "loocv"
            else KFold(n_splits=k, shuffle=True, random_state=seed))


def _eval_linear(
    df: pd.DataFrame,
    kf: KFold,
    grade_levels: List[str],
    model_levels: List[str],
    color_levels: List[str],
) -> Dict[str, float]:
    """
    Cross-validate the plain linear regression model using mileage.

    Returns:
        A dict with keys: {"MAE", "RMSE", "R2", "MAPE"} in JPY units.
    """
    y_true_all, y_pred_all = [], []
    for tr, te in kf.split(df):
        df_tr, df_te = df.iloc[tr], df.iloc[te]
        X_tr, y_tr, gl, ml, cl = _build_design_from_df(df_tr, grade_levels, model_levels, color_levels)

        X_te_raw = df_te[["mileage", "ovrl_grade", "Model", "Color"]].copy()
        X_te_raw["ovrl_grade"] = pd.Categorical(X_te_raw["ovrl_grade"], categories=gl)
        X_te_raw["Model"]      = pd.Categorical(X_te_raw["Model"],      categories=ml)
        X_te_raw["Color"]      = pd.Categorical(X_te_raw["Color"],      categories=cl)
        X_te = pd.get_dummies(X_te_raw, drop_first=True).reindex(columns=X_tr.columns, fill_value=0.0)
        y_te = df_te["final_price"].astype(float).values

        m = LinearRegression().fit(X_tr, y_tr)
        y_pred = m.predict(X_te)

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

    mae  = mean_absolute_error(y_true_all, y_pred_all)
    rmse = sqrt(mean_squared_error(y_true_all, y_pred_all))
    r2   = r2_score(y_true_all, y_pred_all)
    mape = _safe_mape(y_true_all, y_pred_all)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def _eval_logmileage(
    df: pd.DataFrame,
    kf: KFold,
    grade_levels: List[str],
    model_levels: List[str],
    color_levels: List[str],
) -> Dict[str, float]:
    """
    Cross-validate the linear regression model with log(mileage).

    Returns:
        A dict with keys: {"MAE", "RMSE", "R2", "MAPE"} in JPY units.
    """
    y_true_all, y_pred_all = [], []
    for tr, te in kf.split(df):
        df_tr, df_te = df.iloc[tr].copy(), df.iloc[te].copy()

        # Add transformed mileage
        df_tr["log_mileage"] = np.log1p(df_tr["mileage"])
        df_te["log_mileage"] = np.log1p(df_te["mileage"])

        # TRAIN design
        X_tr_raw = df_tr[["log_mileage", "ovrl_grade", "Model", "Color"]].copy()
        X_tr_raw["ovrl_grade"] = pd.Categorical(X_tr_raw["ovrl_grade"], categories=grade_levels)
        X_tr_raw["Model"]      = pd.Categorical(X_tr_raw["Model"],      categories=model_levels)
        X_tr_raw["Color"]      = pd.Categorical(X_tr_raw["Color"],      categories=color_levels)
        X_tr = pd.get_dummies(X_tr_raw, drop_first=True)
        y_tr = df_tr["final_price"].astype(float).values

        # TEST design
        X_te_raw = df_te[["log_mileage", "ovrl_grade", "Model", "Color"]].copy()
        X_te_raw["ovrl_grade"] = pd.Categorical(X_te_raw["ovrl_grade"], categories=grade_levels)
        X_te_raw["Model"]      = pd.Categorical(X_te_raw["Model"],      categories=model_levels)
        X_te_raw["Color"]      = pd.Categorical(X_te_raw["Color"],      categories=color_levels)
        X_te = pd.get_dummies(X_te_raw, drop_first=True).reindex(columns=X_tr.columns, fill_value=0.0)
        y_te = df_te["final_price"].astype(float).values

        m = LinearRegression().fit(X_tr, y_tr)
        y_pred = m.predict(X_te)

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

    mae  = mean_absolute_error(y_true_all, y_pred_all)
    rmse = sqrt(mean_squared_error(y_true_all, y_pred_all))
    r2   = r2_score(y_true_all, y_pred_all)
    mape = _safe_mape(y_true_all, y_pred_all)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def _eval_xgboost(
    df: pd.DataFrame,
    kf: KFold,
    grade_levels: List[str],
    model_levels: List[str],
    color_levels: List[str],
):
    """
    Cross-validate the XGBoost model on the same feature set as the basic model.

    Returns:
        A metrics dict like the other evaluators, or a string if xgboost
        is not installed.
    """
    if not _HAS_XGB:
        return "xgboost not installed"
    y_true_all, y_pred_all = [], []
    for tr, te in kf.split(df):
        df_tr, df_te = df.iloc[tr], df.iloc[te]
        X_tr, y_tr, gl, ml, cl = _build_design_from_df(df_tr, grade_levels, model_levels, color_levels)

        X_te_raw = df_te[["mileage", "ovrl_grade", "Model", "Color"]].copy()
        X_te_raw["ovrl_grade"] = pd.Categorical(X_te_raw["ovrl_grade"], categories=gl)
        X_te_raw["Model"]      = pd.Categorical(X_te_raw["Model"],      categories=ml)
        X_te_raw["Color"]      = pd.Categorical(X_te_raw["Color"],      categories=cl)
        X_te = pd.get_dummies(X_te_raw, drop_first=True).reindex(columns=X_tr.columns, fill_value=0.0)
        y_te = df_te["final_price"].astype(float).values

        m = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, n_jobs=0
        )
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

    mae  = mean_absolute_error(y_true_all, y_pred_all)
    rmse = sqrt(mean_squared_error(y_true_all, y_pred_all))
    r2   = r2_score(y_true_all, y_pred_all)
    mape = _safe_mape(y_true_all, y_pred_all)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def evaluate_models_all(cv: str = "loocv", k: int = 5, random_state: int = 42) -> dict:
    """
    Compare Linear vs Log-Mileage vs XGBoost on identical CV folds.

    Args:
        cv          : "loocv" for leave-one-out, otherwise K-fold.
        k           : Number of folds when cv != "loocv".
        random_state: Seed used when shuffling in K-fold.

    Returns:
        A dict:
            {
                "linear":      {metrics...},
                "log_mileage": {metrics...},
                "xgboost":     {metrics... or "xgboost not installed"},
            }
    """
    df = _fetch_training_df()
    _, _, grade_levels, model_levels, color_levels = _build_design_from_df(df)
    kf = _kfold(df, cv=cv, k=k, seed=random_state)

    results = {
        "linear":      _eval_linear(df, kf, grade_levels, model_levels, color_levels),
        "log_mileage": _eval_logmileage(df, kf, grade_levels, model_levels, color_levels),
        "xgboost":     _eval_xgboost(df, kf, grade_levels, model_levels, color_levels),
    }
    return results


def print_eval_table(results: dict, decimals: int = 2):
    """
    Pretty-print evaluation results as a comparison table.

    Args:
        results : Output from evaluate_models_all(...) or similar.
        decimals: Number of decimal places to show for each metric.
    """
    # Collect all metric names
    metrics = list(next(iter(results.values())).keys())

    # Header
    header = ["Model"] + metrics
    rows = []
    for model_name, scores in results.items():
        row = [model_name]
        for m in metrics:
            val = scores[m]
            row.append(f"{val:.{decimals}f}")
        rows.append(row)

    # Compute column widths
    col_widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]

    # Print header
    header_str = " | ".join(str(h).ljust(w) for h, w in zip(header, col_widths))
    print(header_str)
    print("-+-".join("-" * w for w in col_widths))

    # Print rows
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


# ------------------------------
# (Optional) Quick CLI sanity checks
# ------------------------------
if __name__ == "__main__":
    # Basic sanity check when run as a script:
    # - try training models
    # - run a few example predictions
    try:
        info_lin = train_basic(min_rows=6)
        print("[train_basic] ->", info_lin)
    except Exception as e:
        print("[train_basic] skipped:", e)

    if _HAS_XGB:
        try:
            info_xgb = train_xgb(min_rows=6)
            print("[train_xgb] ->", info_xgb)
        except Exception as e:
            print("[train_xgb] skipped:", e)

    try:
        ex1 = {"Model": "E400",    "mileage": 63000, "ovrl_grade": "4.5", "colour": "Any"}
        ex2 = {"Model": "E43",     "mileage": 63000, "ovrl_grade": "4.5", "colour": "Black"}
        ex3 = {"Model": "E53_PRE", "mileage": 63000, "ovrl_grade": "4.5", "colour": "White"}
        ex4 = {"Model": "E53_FL",  "mileage": 63000, "ovrl_grade": "4.5", "colour": "Blue"}

        print("LIN  E400 (Any)   :", predict_basic(ex1))
        print("LIN  E43  (Black) :", predict_basic(ex2))
        print("LIN  E53_PRE      :", predict_basic(ex3))
        print("LIN  E53_FL       :", predict_basic(ex4))

        if _HAS_XGB and MODEL_PATH_XGB.exists():
            print("XGB  E400 (Any)   :", predict_xgb(ex1))
            print("XGB  E43  (Black) :", predict_xgb(ex2))
            print("XGB  E53_PRE      :", predict_xgb(ex3))
            print("XGB  E53_FL       :", predict_xgb(ex4))
    except Exception as e:
        print("[predict] error:", e)

