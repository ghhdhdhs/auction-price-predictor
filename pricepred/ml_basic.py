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
    • Estimating prediction intervals
          - Linear/logm: empirical residual quantiles (q05/q95)
          - XGB: OOF conformal calibration (absolute-error quantile), stored once at training time
    • Running model inference with consistent preprocessing
    • Cross-validated evaluation (MAE / RMSE / R² / MAPE)

Notes:
- Prediction-time CI for XGB is O(1): low=yhat-q, high=yhat+q (q is precomputed at training).
- Final XGB model is still trained on ALL rows (no training data loss).
- This module is Django-aware only through importing CarSale in _fetch_training_df().
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re

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

# Django model (the only Django dependency in this module)
from .models import CarSale


# ------------------------------
# Paths
# ------------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH_LIN = MODEL_DIR / "basic_model.pkl"        # Linear regression (raw mileage)
MODEL_PATH_LOGM = MODEL_DIR / "logm_model.pkl"        # Linear regression (log-mileage)
MODEL_PATH_XGB = MODEL_DIR / "xgb_model.pkl"          # XGBoost


# ------------------------------
# Colour policy (UI + averaging)
# ------------------------------
COLOR_ANY = "__ANY__"
COLOR_OTHER_BUCKET = "__OTHER_BUCKET__"

# Colours shown as explicit choices in the UI (stable / interpretable)
UI_MAIN_COLORS = {"White", "Black", "Silver"}


# ------------------------------
# Helpers
# ------------------------------
def _parse_year(year_raw: Any) -> Optional[int]:
    """
    Parse a 4-digit year from various raw formats.
    Returns int year if found, otherwise None.
    """
    if year_raw is None or (isinstance(year_raw, float) and pd.isna(year_raw)):
        return None
    try:
        return int(year_raw)
    except Exception:
        pass

    m = re.search(r"\d{4}", str(year_raw))
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def _normalize_color_from_db(series: pd.Series) -> pd.Series:
    """
    Normalise DB color into canonical labels:
        White, Black, Silver, Blue, Gray, Red, Other
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

    Returns DataFrame columns:
        mileage (float)
        ovrl_grade (str)
        Model (str): E400 / E43 / E53_PRE / E53_FL
        Color (str): White/Black/Silver/Blue/Gray/Red/Other
        final_price (float)
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

    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
    df["final_price"] = pd.to_numeric(df["final_price"], errors="coerce")

    def _norm_grade(v) -> Optional[str]:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip().upper()
        if not s:
            return None
        try:
            return f"{float(s):.1f}"
        except Exception:
            return s

    df["ovrl_grade"] = df["auction_grade"].map(_norm_grade)

    def _model_from_row(row) -> Optional[str]:
        s = (row.get("chassis_code__variant") or "").upper()
        year_int = _parse_year(row.get("year"))

        if "E53" in s:
            if year_int is not None and year_int >= 2021:
                return "E53_FL"
            return "E53_PRE"
        if "E43" in s:
            return "E43"
        if "E400" in s:
            return "E400"
        return None

    df["Model"] = df.apply(_model_from_row, axis=1)
    df["Color"] = _normalize_color_from_db(df["color"])

    df = df.dropna(subset=["mileage", "final_price", "ovrl_grade", "Model", "Color"])
    if df.empty:
        raise ValueError("No usable rows after filtering for mileage/grade/Model/Color.")

    return df[["mileage", "ovrl_grade", "Model", "Color", "final_price"]]


def _freeze_and_dummify(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str], List[str]]:
    """
    One-hot encode the main features and freeze categorical levels.
    """
    grade_levels: List[str] = sorted(df["ovrl_grade"].unique().tolist())
    model_levels: List[str] = sorted(df["Model"].unique().tolist())
    color_levels: List[str] = sorted(df["Color"].unique().tolist())

    if "Other" not in color_levels:
        color_levels.append("Other")

    X_raw = df[["mileage", "ovrl_grade", "Model", "Color"]].copy()
    X_raw["ovrl_grade"] = pd.Categorical(X_raw["ovrl_grade"], categories=grade_levels, ordered=False)
    X_raw["Model"]      = pd.Categorical(X_raw["Model"],      categories=model_levels, ordered=False)
    X_raw["Color"]      = pd.Categorical(X_raw["Color"],      categories=color_levels, ordered=False)

    X = pd.get_dummies(X_raw, drop_first=True)
    y = df["final_price"].astype(float).to_numpy()
    return X, y, grade_levels, model_levels, color_levels


def _normalize_grade_for_payload(raw_grade: Any) -> str:
    if raw_grade is None:
        raise ValueError("ovrl_grade is required")
    s = str(raw_grade).strip().upper()
    if not s:
        raise ValueError("ovrl_grade is empty")
    try:
        return f"{float(s):.1f}"
    except Exception:
        return s


def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        mileage = float(payload["mileage"])
    except Exception as e:
        raise ValueError(f"Invalid mileage in payload: {payload.get('mileage')}") from e

    grade_str = _normalize_grade_for_payload(payload.get("ovrl_grade"))
    model_str = str(payload["Model"]).upper().strip()

    return {"mileage": mileage, "ovrl_grade": grade_str, "Model": model_str}


def _normalize_color_for_payload(
    payload: Dict[str, Any],
    color_levels: List[str],
    default_color: Optional[str],
) -> str:
    """
    Colour selection policy:
      - missing / blank / Any => COLOR_ANY (market-average)
      - "Other" or unknown literal => COLOR_OTHER_BUCKET (avg of non-main colours)
      - specific => canonical label (White/Black/Silver/Blue/Gray/Red/Other)
    """
    raw = payload.get("colour") or payload.get("color")
    if raw is None:
        return COLOR_ANY

    txt = str(raw).strip()
    up = txt.upper()

    if up == "" or up in {"ALL", "ANY", "ANY COLOUR", "ANY COLOR", "ALL COLOURS", "ALL COLORS"}:
        return COLOR_ANY
    if up in {"OTHER", "OTHERS"}:
        return COLOR_OTHER_BUCKET

    if "WHITE" in up:
        cand = "White"
    elif "BLACK" in up:
        cand = "Black"
    elif "SILVER" in up:
        cand = "Silver"
    elif "BLUE" in up or "NAVY" in up:
        cand = "Blue"
    elif "GRAY" in up or "GREY" in up:
        cand = "Gray"
    elif "RED" in up:
        cand = "Red"
    else:
        return COLOR_OTHER_BUCKET

    if cand not in color_levels:
        if "Other" in color_levels:
            return "Other"
        if default_color is not None:
            return default_color
        return color_levels[0] if color_levels else "Other"

    return cand


def _weights_for_color_choice(
    choice: str,
    color_levels: List[str],
    priors: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    """
    Build a normalised weight vector over colours for:
      - Any (all colours, weighted by training frequency)
      - Other bucket (all non-main colours, weighted)
      - Specific colour (degenerate weight=1)
    """
    if not isinstance(priors, dict) or not priors:
        return None

    levels = [c for c in color_levels if c in priors]

    if choice == COLOR_ANY:
        allowed = set(levels)
    elif choice == COLOR_OTHER_BUCKET:
        allowed = set(levels) - UI_MAIN_COLORS
    else:
        allowed = {choice}

    allowed = {c for c in allowed if c in color_levels}
    weights = {c: float(priors.get(c, 0.0)) for c in allowed}
    total = sum(weights.values())

    if total <= 0 or not weights:
        allowed_list = sorted(list(allowed))
        if not allowed_list:
            return None
        u = 1.0 / len(allowed_list)
        return {c: u for c in allowed_list}

    return {c: w / total for c, w in weights.items()}


def _row_to_X(
    norm: Dict[str, Any],
    cols: List[str],
    grades: List[str],
    models: List[str],
    colors: List[str],
) -> pd.DataFrame:
    """
    Build a one-row design matrix aligned with the training feature columns.
    """
    row = pd.DataFrame([norm])
    row["ovrl_grade"] = pd.Categorical(row["ovrl_grade"], categories=grades, ordered=False)
    row["Model"]      = pd.Categorical(row["Model"],      categories=models, ordered=False)
    row["Color"]      = pd.Categorical(row["Color"],      categories=colors, ordered=False)
    X = pd.get_dummies(row, drop_first=True).reindex(columns=cols, fill_value=0.0)
    return X


# ------------------------------
# Residual → CI helper (linear/logm fallback)
# ------------------------------
def _compute_ci_from_residuals(
    norm: Dict[str, Any],
    bundle: Dict[str, Any],
    y_hat: float,
    fallback_band_jpy: float = 50000.0,
    fallback_pct: float = 0.15,
) -> Tuple[float, float]:
    """
    Compute an approximate 90% prediction interval using stored residual quantiles.
    """
    resid_meta = bundle.get("residuals")
    if not isinstance(resid_meta, dict):
        band = max(fallback_pct * abs(y_hat), fallback_band_jpy)
        return max(y_hat - band, 0.0), max(y_hat + band, 0.0)

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
        band = max(fallback_pct * abs(y_hat), fallback_band_jpy)
        return max(y_hat - band, 0.0), max(y_hat + band, 0.0)

    return max(y_hat + float(q05), 0.0), max(y_hat + float(q95), 0.0)


# ------------------------------
# Linear Regression (basic)
# ------------------------------
def train_basic(min_rows: int = 6) -> Dict[str, Any]:
    df = _fetch_training_df()
    if len(df) < min_rows:
        raise ValueError(f"Need ≥{min_rows} rows, have {len(df)}.")

    X, y, grade_levels, model_levels, color_levels = _freeze_and_dummify(df)

    default_color = df["Color"].value_counts().idxmax()
    color_priors = df["Color"].value_counts(normalize=True).to_dict()
    if "Other" not in color_priors:
        color_priors["Other"] = 0.0

    model = LinearRegression().fit(X, y)

    y_hat = model.predict(X)
    df["_resid"] = y - y_hat

    global_q05, global_q95 = np.quantile(df["_resid"], [0.05, 0.95])

    resid_by_mg: Dict[str, Tuple[float, float]] = {}
    for (m, g), grp in df.groupby(["Model", "ovrl_grade"]):
        if len(grp) < max(8, min_rows):
            continue
        q05, q95 = np.quantile(grp["_resid"], [0.05, 0.95])
        resid_by_mg[f"{m}||{g}"] = (float(q05), float(q95))

    resid_by_m: Dict[str, Tuple[float, float]] = {}
    for m, grp in df.groupby("Model"):
        if len(grp) < max(8, min_rows):
            continue
        q05, q95 = np.quantile(grp["_resid"], [0.05, 0.95])
        resid_by_m[str(m)] = (float(q05), float(q95))

    bundle = {
        "model": model,
        "columns": X.columns.tolist(),
        "grade_levels": grade_levels,
        "model_levels": model_levels,
        "color_levels": color_levels,
        "meta": {
            "n_rows": int(len(df)),
            "algorithm": "linear",
            "default_color": default_color,
            "color_priors": color_priors,
            "ci_method": "residual_q05_q95",
            "ci_level": 0.90,
        },
        "residuals": {
            "global_q05": float(global_q05),
            "global_q95": float(global_q95),
            "by_model_grade": resid_by_mg,
            "by_model": resid_by_m,
        },
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


def predict_basic_with_ci(payload: Dict[str, Any]) -> Tuple[float, float, float]:
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
    priors: Optional[Dict[str, float]] = meta.get("color_priors")

    if norm["ovrl_grade"] not in grades:
        raise ValueError(f"ovrl_grade '{norm['ovrl_grade']}' was not seen in training. Seen: {grades}")
    if norm["Model"] not in models:
        raise ValueError(f"Model '{norm['Model']}' was not seen in training. Seen: {models}")

    color_choice = _normalize_color_for_payload(payload, colors, default_color)
    model = bundle["model"]

    weights = _weights_for_color_choice(color_choice, colors, priors)

    if weights is not None and color_choice in {COLOR_ANY, COLOR_OTHER_BUCKET}:
        y_hat = 0.0
        for c, w in weights.items():
            n2 = dict(norm)
            n2["Color"] = c
            Xc = _row_to_X(n2, cols, grades, models, colors)
            y_hat += w * float(model.predict(Xc)[0])
        norm["Color"] = default_color or (colors[0] if colors else "Other")
    else:
        if color_choice in {COLOR_ANY, COLOR_OTHER_BUCKET}:
            norm["Color"] = default_color or (colors[0] if colors else "Other")
        else:
            norm["Color"] = color_choice

        if norm["Color"] not in colors:
            raise ValueError(f"Color '{norm['Color']}' was not seen in training. Seen: {colors}")

        X = _row_to_X(norm, cols, grades, models, colors)
        y_hat = float(model.predict(X)[0])

    low, high = _compute_ci_from_residuals(norm, bundle, y_hat)
    return float(y_hat), float(low), float(high)


def predict_basic(payload: Dict[str, Any]) -> float:
    y_hat, _, _ = predict_basic_with_ci(payload)
    return float(y_hat)


# ------------------------------
# Linear Regression with log-mileage
# ------------------------------
def train_logm(min_rows: int = 6) -> Dict[str, Any]:
    df = _fetch_training_df()
    if len(df) < min_rows:
        raise ValueError(f"Need ≥{min_rows} rows, have {len(df)}.")

    df = df.copy()
    df["log_mileage"] = np.log1p(df["mileage"])

    grade_levels: List[str] = sorted(df["ovrl_grade"].unique().tolist())
    model_levels: List[str] = sorted(df["Model"].unique().tolist())
    color_levels: List[str] = sorted(df["Color"].unique().tolist())
    if "Other" not in color_levels:
        color_levels.append("Other")

    default_color = df["Color"].value_counts().idxmax()
    color_priors = df["Color"].value_counts(normalize=True).to_dict()
    if "Other" not in color_priors:
        color_priors["Other"] = 0.0

    X_raw = df[["log_mileage", "ovrl_grade", "Model", "Color"]].copy()
    X_raw["ovrl_grade"] = pd.Categorical(X_raw["ovrl_grade"], categories=grade_levels, ordered=False)
    X_raw["Model"]      = pd.Categorical(X_raw["Model"],      categories=model_levels, ordered=False)
    X_raw["Color"]      = pd.Categorical(X_raw["Color"],      categories=color_levels, ordered=False)

    X = pd.get_dummies(X_raw, drop_first=True)
    y = df["final_price"].astype(float).to_numpy()

    model = LinearRegression().fit(X, y)

    y_hat = model.predict(X)
    df["_resid"] = y - y_hat
    global_q05, global_q95 = np.quantile(df["_resid"], [0.05, 0.95])

    resid_by_mg: Dict[str, Tuple[float, float]] = {}
    for (m, g), grp in df.groupby(["Model", "ovrl_grade"]):
        if len(grp) < max(8, min_rows):
            continue
        q05, q95 = np.quantile(grp["_resid"], [0.05, 0.95])
        resid_by_mg[f"{m}||{g}"] = (float(q05), float(q95))

    resid_by_m: Dict[str, Tuple[float, float]] = {}
    for m, grp in df.groupby("Model"):
        if len(grp) < max(8, min_rows):
            continue
        q05, q95 = np.quantile(grp["_resid"], [0.05, 0.95])
        resid_by_m[str(m)] = (float(q05), float(q95))

    bundle = {
        "model": model,
        "columns": X.columns.tolist(),
        "grade_levels": grade_levels,
        "model_levels": model_levels,
        "color_levels": color_levels,
        "meta": {
            "n_rows": int(len(df)),
            "algorithm": "logm_linear",
            "transform": "log_mileage",
            "default_color": default_color,
            "color_priors": color_priors,
            "ci_method": "residual_q05_q95",
            "ci_level": 0.90,
        },
        "residuals": {
            "global_q05": float(global_q05),
            "global_q95": float(global_q95),
            "by_model_grade": resid_by_mg,
            "by_model": resid_by_m,
        },
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


def predict_logm_with_ci(payload: Dict[str, Any]) -> Tuple[float, float, float]:
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
    priors: Optional[Dict[str, float]] = meta.get("color_priors")

    if norm["ovrl_grade"] not in grades:
        raise ValueError(f"ovrl_grade '{norm['ovrl_grade']}' was not seen in training. Seen: {grades}")
    if norm["Model"] not in models:
        raise ValueError(f"Model '{norm['Model']}' was not seen in training. Seen: {models}")

    color_choice = _normalize_color_for_payload(payload, colors, default_color)
    model = bundle["model"]

    def _pred_for_color(c: str) -> float:
        row = pd.DataFrame([{
            "log_mileage": np.log1p(norm["mileage"]),
            "ovrl_grade": norm["ovrl_grade"],
            "Model": norm["Model"],
            "Color": c,
        }])
        row["ovrl_grade"] = pd.Categorical(row["ovrl_grade"], categories=grades)
        row["Model"]      = pd.Categorical(row["Model"],      categories=models)
        row["Color"]      = pd.Categorical(row["Color"],      categories=colors)
        Xc = pd.get_dummies(row, drop_first=True).reindex(columns=cols, fill_value=0.0)
        return float(model.predict(Xc)[0])

    weights = _weights_for_color_choice(color_choice, colors, priors)

    if weights is not None and color_choice in {COLOR_ANY, COLOR_OTHER_BUCKET}:
        y_hat = sum(w * _pred_for_color(c) for c, w in weights.items())
        norm["Color"] = default_color or (colors[0] if colors else "Other")
    else:
        if color_choice in {COLOR_ANY, COLOR_OTHER_BUCKET}:
            norm["Color"] = default_color or (colors[0] if colors else "Other")
        else:
            norm["Color"] = color_choice

        if norm["Color"] not in colors:
            raise ValueError(f"Color '{norm['Color']}' was not seen in training. Seen: {colors}")

        y_hat = _pred_for_color(norm["Color"])

    low, high = _compute_ci_from_residuals(norm, bundle, y_hat)
    return float(y_hat), float(low), float(high)


def predict_logm(payload: Dict[str, Any]) -> float:
    y_hat, _, _ = predict_logm_with_ci(payload)
    return float(y_hat)


# ------------------------------
# XGBoost + Conformal CI
# ------------------------------
def _make_xgb_regressor(random_state: int = 42) -> "xgb.XGBRegressor":
    """
    Single source of truth for XGB hyperparams.
    """
    return xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=0,
    )


def _xgb_oof_conformal_q(
    X: pd.DataFrame,
    y: np.ndarray,
    model_codes: np.ndarray,
    *,
    ci_level: float = 0.90,
    k: int = 5,
    random_state: int = 42,
    min_group: int = 8,
) -> Tuple[float, Dict[str, float]]:
    """
    Out-of-fold conformal calibration:
      q = quantile_{ci_level}(|y - y_oof|)

    Returns:
      q_global, q_by_model
    """
    from sklearn.model_selection import KFold

    n = len(y)
    if n < max(10, k):
        m = _make_xgb_regressor(random_state=random_state)
        m.fit(X, y)
        abs_err = np.abs(y - m.predict(X))
        q_global = float(np.quantile(abs_err, ci_level))
        return q_global, {}

    k = min(k, n)
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    oof = np.empty(n, dtype=float)
    for tr_idx, va_idx in kf.split(X):
        m = _make_xgb_regressor(random_state=random_state)
        m.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(X.iloc[va_idx])

    abs_err = np.abs(y - oof)
    q_global = float(np.quantile(abs_err, ci_level))

    q_by_model: Dict[str, float] = {}
    for mcode in np.unique(model_codes):
        mask = (model_codes == mcode)
        if int(mask.sum()) >= int(min_group):
            q_by_model[str(mcode)] = float(np.quantile(abs_err[mask], ci_level))

    return q_global, q_by_model


def train_xgb(
    min_rows: int = 6,
    random_state: int = 42,
    ci_level: float = 0.90,
    oof_k: int = 5,
) -> Dict[str, Any]:
    """
    Train an XGBoost regressor on the same feature set as the basic model.

    CI (FIXED):
      - Compute OOF conformal width q (absolute error quantile) once during training
      - Store q globally and per Model (if enough rows)
      - Final model still fits on ALL rows (no training data lost)
      - Prediction-time CI is O(1)
    """
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not installed. Run: pip install xgboost")

    df = _fetch_training_df()
    if len(df) < min_rows:
        raise ValueError(f"Need ≥{min_rows} rows, have {len(df)}.")

    X, y, grade_levels, model_levels, color_levels = _freeze_and_dummify(df)

    default_color = df["Color"].value_counts().idxmax()
    color_priors = df["Color"].value_counts(normalize=True).to_dict()
    if "Other" not in color_priors:
        color_priors["Other"] = 0.0

    # Conformal calibration from OOF predictions
    model_codes = df["Model"].astype(str).to_numpy()
    q_global, q_by_model = _xgb_oof_conformal_q(
        X,
        y,
        model_codes,
        ci_level=ci_level,
        k=oof_k,
        random_state=random_state,
        min_group=max(8, min_rows),
    )

    # Fit final model on ALL data
    model = _make_xgb_regressor(random_state=random_state)
    model.fit(X, y)

    # Keep residual quantiles as fallback (for old bundles or if q is missing)
    y_hat = model.predict(X)
    df["_resid"] = y - y_hat
    global_q05, global_q95 = np.quantile(df["_resid"], [0.05, 0.95])

    resid_by_mg: Dict[str, Tuple[float, float]] = {}
    for (m, g), grp in df.groupby(["Model", "ovrl_grade"]):
        if len(grp) < max(8, min_rows):
            continue
        q05, q95 = np.quantile(grp["_resid"], [0.05, 0.95])
        resid_by_mg[f"{m}||{g}"] = (float(q05), float(q95))

    resid_by_m: Dict[str, Tuple[float, float]] = {}
    for m, grp in df.groupby("Model"):
        if len(grp) < max(8, min_rows):
            continue
        q05, q95 = np.quantile(grp["_resid"], [0.05, 0.95])
        resid_by_m[str(m)] = (float(q05), float(q95))

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
            "color_priors": color_priors,

            # NEW: conformal CI metadata
            "ci_method": "conformal_oof_abs",
            "ci_level": float(ci_level),
            "xgb_q_abs_global": float(q_global),
            "xgb_q_abs_by_model": q_by_model,
            "oof_k": int(oof_k),
        },
        "residuals": {
            "global_q05": float(global_q05),
            "global_q95": float(global_q95),
            "by_model_grade": resid_by_mg,
            "by_model": resid_by_m,
        },
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
        "ci_method": "conformal_oof_abs",
        "ci_level": float(ci_level),
        "xgb_q_abs_global": float(q_global),
    }


def predict_xgb_with_ci(payload: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Predict using XGB and return CI.

    CI (FAST):
      - Prefer conformal OOF q from bundle meta:
            low = yhat - q
            high = yhat + q
        where q optionally differs per Model (variant)
      - If missing, fall back to residual-based q05/q95
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
    priors: Optional[Dict[str, float]] = meta.get("color_priors")

    if norm["ovrl_grade"] not in grades:
        raise ValueError(f"ovrl_grade '{norm['ovrl_grade']}' was not seen in training. Seen: {grades}")
    if norm["Model"] not in models:
        raise ValueError(f"Model '{norm['Model']}' was not seen in training. Seen: {models}")

    color_choice = _normalize_color_for_payload(payload, colors, default_color)
    model = bundle["model"]

    # Point prediction (same behaviour as linear/logm)
    weights = _weights_for_color_choice(color_choice, colors, priors)

    if weights is not None and color_choice in {COLOR_ANY, COLOR_OTHER_BUCKET}:
        y_hat = 0.0
        for c, w in weights.items():
            n2 = dict(norm)
            n2["Color"] = c
            Xc = _row_to_X(n2, cols, grades, models, colors)
            y_hat += w * float(model.predict(Xc)[0])
        norm["Color"] = default_color or (colors[0] if colors else "Other")
    else:
        if color_choice in {COLOR_ANY, COLOR_OTHER_BUCKET}:
            norm["Color"] = default_color or (colors[0] if colors else "Other")
        else:
            norm["Color"] = color_choice

        if norm["Color"] not in colors:
            raise ValueError(f"Color '{norm['Color']}' was not seen in training. Seen: {colors}")

        X = _row_to_X(norm, cols, grades, models, colors)
        y_hat = float(model.predict(X)[0])

    # CI: conformal q if available, else residual fallback
    if meta.get("ci_method") == "conformal_oof_abs":
        q_global = float(meta.get("xgb_q_abs_global", 0.0) or 0.0)
        q_by_model = meta.get("xgb_q_abs_by_model") or {}
        q = float(q_by_model.get(norm["Model"], q_global))

        if q > 0:
            low = max(float(y_hat - q), 0.0)
            high = max(float(y_hat + q), 0.0)
            return float(y_hat), float(low), float(high)

    low, high = _compute_ci_from_residuals(norm, bundle, y_hat)
    return float(y_hat), float(low), float(high)


def predict_xgb(payload: Dict[str, Any]) -> float:
    y_hat, _, _ = predict_xgb_with_ci(payload)
    return float(y_hat)


# =========================
# Model evaluation helpers
# =========================
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_mape(y_true, y_pred, eps: float = 1e-9) -> float:
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
    if grade_levels is None:
        grade_levels = sorted(df["ovrl_grade"].unique().tolist())
    if model_levels is None:
        model_levels = sorted(df["Model"].unique().tolist())
    if color_levels is None:
        color_levels = sorted(df["Color"].unique().tolist())
        if "Other" not in color_levels:
            color_levels.append("Other")

    X_raw = df[["mileage", "ovrl_grade", "Model", "Color"]].copy()
    X_raw["ovrl_grade"] = pd.Categorical(X_raw["ovrl_grade"], categories=grade_levels, ordered=False)
    X_raw["Model"]      = pd.Categorical(X_raw["Model"],      categories=model_levels, ordered=False)
    X_raw["Color"]      = pd.Categorical(X_raw["Color"],      categories=color_levels, ordered=False)

    X = pd.get_dummies(X_raw, drop_first=True)
    y = df["final_price"].astype(float).values
    return X, y, grade_levels, model_levels, color_levels


def _kfold(df: pd.DataFrame, cv: str = "loocv", k: int = 5, seed: int = 42) -> KFold:
    return (
        KFold(n_splits=len(df), shuffle=False)
        if cv.lower() == "loocv"
        else KFold(n_splits=k, shuffle=True, random_state=seed)
    )


def _eval_linear(df: pd.DataFrame, kf: KFold, grade_levels: List[str], model_levels: List[str], color_levels: List[str]) -> Dict[str, float]:
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


def _eval_logmileage(df: pd.DataFrame, kf: KFold, grade_levels: List[str], model_levels: List[str], color_levels: List[str]) -> Dict[str, float]:
    y_true_all, y_pred_all = [], []
    for tr, te in kf.split(df):
        df_tr, df_te = df.iloc[tr].copy(), df.iloc[te].copy()
        df_tr["log_mileage"] = np.log1p(df_tr["mileage"])
        df_te["log_mileage"] = np.log1p(df_te["mileage"])

        X_tr_raw = df_tr[["log_mileage", "ovrl_grade", "Model", "Color"]].copy()
        X_tr_raw["ovrl_grade"] = pd.Categorical(X_tr_raw["ovrl_grade"], categories=grade_levels)
        X_tr_raw["Model"]      = pd.Categorical(X_tr_raw["Model"],      categories=model_levels)
        X_tr_raw["Color"]      = pd.Categorical(X_tr_raw["Color"],      categories=color_levels)
        X_tr = pd.get_dummies(X_tr_raw, drop_first=True)
        y_tr = df_tr["final_price"].astype(float).values

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


def _eval_xgboost(df: pd.DataFrame, kf: KFold, grade_levels: List[str], model_levels: List[str], color_levels: List[str]):
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

        m = _make_xgb_regressor(random_state=42)
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
    df = _fetch_training_df()
    _, _, grade_levels, model_levels, color_levels = _build_design_from_df(df)
    kf = _kfold(df, cv=cv, k=k, seed=random_state)

    return {
        "linear":      _eval_linear(df, kf, grade_levels, model_levels, color_levels),
        "log_mileage": _eval_logmileage(df, kf, grade_levels, model_levels, color_levels),
        "xgboost":     _eval_xgboost(df, kf, grade_levels, model_levels, color_levels),
    }


def print_eval_table(results: dict, decimals: int = 2):
    metrics = None
    for v in results.values():
        if isinstance(v, dict):
            metrics = list(v.keys())
            break

    if not metrics:
        for k, v in results.items():
            print(f"{k}: {v}")
        return

    header = ["Model"] + metrics
    rows = []

    for model_name, scores in results.items():
        row = [model_name]
        if isinstance(scores, dict):
            for m in metrics:
                val = scores.get(m)
                if isinstance(val, (int, float)):
                    row.append(f"{val:.{decimals}f}")
                else:
                    row.append(str(val))
        else:
            row.extend([str(scores)] + [""] * (len(metrics) - 1))
        rows.append(row)

    col_widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]

    header_str = " | ".join(str(h).ljust(w) for h, w in zip(header, col_widths))
    print(header_str)
    print("-+-".join("-" * w for w in col_widths))

    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


# ------------------------------
# Optional CLI sanity checks
# ------------------------------
if __name__ == "__main__":
    try:
        info_lin = train_basic(min_rows=6)
        print("[train_basic] ->", info_lin)
    except Exception as e:
        print("[train_basic] skipped:", e)

    try:
        info_logm = train_logm(min_rows=6)
        print("[train_logm] ->", info_logm)
    except Exception as e:
        print("[train_logm] skipped:", e)

    if _HAS_XGB:
        try:
            info_xgb = train_xgb(min_rows=6, oof_k=5, ci_level=0.90)
            print("[train_xgb] ->", info_xgb)
        except Exception as e:
            print("[train_xgb] skipped:", e)

    try:
        ex1 = {"Model": "E400",    "mileage": 63000, "ovrl_grade": "4.5", "colour": "Any"}
        ex2 = {"Model": "E43",     "mileage": 63000, "ovrl_grade": "4.5", "colour": "Black"}
        ex3 = {"Model": "E53_PRE", "mileage": 63000, "ovrl_grade": "4.5", "colour": "White"}
        ex4 = {"Model": "E53_FL",  "mileage": 63000, "ovrl_grade": "4.5", "colour": "Other"}

        print("LIN  E400 (Any/avg) :", predict_basic_with_ci(ex1))
        print("LOGM E400 (Any/avg) :", predict_logm_with_ci(ex1))

        if _HAS_XGB and MODEL_PATH_XGB.exists():
            print("XGB  E400 (Any/avg) :", predict_xgb_with_ci(ex1))
            print("XGB  E43  (Black)   :", predict_xgb_with_ci(ex2))
            print("XGB  E53_PRE (White):", predict_xgb_with_ci(ex3))
            print("XGB  E53_FL (Other) :", predict_xgb_with_ci(ex4))
    except Exception as e:
        print("[predict] error:", e)
