"""
Django views for the Price Prediction Dashboard.

This module contains three categories of views:

1. --------------------------------------------------------------------------
   HTML VIEWS (server-rendered)
   --------------------------------------------------------------------------
   • add_car_sale
        Form-driven creation of CarSale records. Used primarily for internal
        data entry and testing during model development.

   • dashboard
        Main user interface for the prediction tool. Displays:
            - A prediction form (mileage, model, grade, colour)
            - A scatter plot of historical auction data (fetched via API)
        The selectable model and grade options are generated from the same
        cleaned training DataFrame used by the ML models, ensuring that all
        dropdowns remain in sync with the actual training schema.

2. --------------------------------------------------------------------------
   HTMX / PARTIALS
   --------------------------------------------------------------------------
   • chassis_options
        Small HTML fragment endpoint returning <option> elements for a chassis
        dropdown based on the selected maker. This keeps the form dynamic
        without needing full page reloads.

3. --------------------------------------------------------------------------
   JSON API ENDPOINTS (used by the frontend JavaScript)
   --------------------------------------------------------------------------
   • api_predict
        Core inference endpoint used by the dashboard prediction form.
        Accepts JSON with:
            - mileage
            - ovrl_grade
            - Model        (internal label: 'E400', 'E43', 'E53_PRE', 'E53_FL')
            - colour/color (optional)
            - model_type   (optional: "linear", "logm", "xgb")
        Returns:
            - rounded point estimate in JPY
            - approximate 90% prediction interval (low/high)
            - model_used and CI metadata

   • api_history
        Returns historical auction points (mileage, price, model, grade)
        for plotting the scatter chart on the dashboard. Uses the same data
        pipeline as ML training to guarantee consistency across the app.

------------------------------------------------------------------------------
Implementation Notes
------------------------------------------------------------------------------
• All ML predictions funnel through ml_basic.py (predict_*_with_ci), which
  provides consistent residual-based confidence intervals.

• _fetch_training_df() is used both by the ML layer and by the dashboard,
  ensuring that the user interface reflects the true available training
  categories.

• Errors are intentionally returned as HTTP 400 with short, user-facing
  messages to avoid exposing stack traces in production.

• The views are structured to keep the Django layer extremely thin:
  they validate inputs, call the ML module, and return formatted responses.

------------------------------------------------------------------------------
"""

from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import json

from .forms import CarSaleForm
from .models import Chassis
from .ml_basic import (
    _fetch_training_df,
    predict_basic_with_ci,
    predict_logm_with_ci,
    predict_xgb_with_ci,
)


@require_http_methods(["GET", "POST"])
def add_car_sale(request):
    if request.method == "POST":
        form = CarSaleForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("/admin/")  # or a success page
    else:
        form = CarSaleForm()
    return render(request, "add_sale.html", {"form": form})


# HTMX endpoint: returns <option> list for chassis based on maker
@require_http_methods(["GET"])
def chassis_options(request):
    maker_id = request.GET.get("maker_id")
    options = (
        Chassis.objects.filter(maker_id=maker_id).values_list("id", "code")
        if maker_id
        else []
    )
    html = "".join(f'<option value="{cid}">{code}</option>' for cid, code in options)
    return HttpResponse(html or '<option value="">-- select maker first --</option>')


# -------------------------
# API: predict auction price
# -------------------------
@csrf_exempt
@require_http_methods(["POST"])
def api_predict(request):
    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return HttpResponseBadRequest("Invalid JSON")

    required = ["mileage", "ovrl_grade", "Model"]
    missing = [k for k in required if k not in data or data[k] in ("", None)]
    if missing:
        return HttpResponseBadRequest(f"Missing fields: {', '.join(missing)}")

    # Which ML model to use:
    #   "logm"   -> log-mileage linear (default)
    #   "linear" -> plain linear
    #   "xgb"    -> XGBoost
    model_type = (data.get("model_type") or "logm").lower()

    try:
        if model_type == "linear":
            yhat, low, high = predict_basic_with_ci(data)
            model_used = "linear"
        elif model_type == "xgb":
            yhat, low, high = predict_xgb_with_ci(data)
            model_used = "xgb"
        else:
            # default: log-mileage model
            yhat, low, high = predict_logm_with_ci(data)
            model_used = "logm"

    except FileNotFoundError as e:
        # Raised if the corresponding model bundle file doesn't exist
        return HttpResponseBadRequest(str(e))
    except RuntimeError as e:
        # e.g. xgboost not installed
        return HttpResponseBadRequest(str(e))
    except Exception as e:
        return HttpResponseBadRequest(str(e))

    # yhat, low, high are already based on the regression residuals (≈90% CI)
    return JsonResponse(
        {
            "price_jpy": round(yhat),
            "low_jpy": round(low),
            "high_jpy": round(high),
            "model_used": model_used,
            "ci_level": 0.90,
        }
    )


@require_http_methods(["GET"])
def dashboard(request):
    """
    Main page: shows history scatter plot + prediction form.

    We derive model_options and grade_options from the same
    training DataFrame used by the ML model so everything
    stays in sync (E53_PRE / E53_FL included).
    """
    try:
        df = _fetch_training_df()  # mileage, ovrl_grade, Model, final_price

        # unique model codes used in training, e.g.:
        # ['E400', 'E43', 'E53_PRE', 'E53_FL']
        model_options = sorted(df["Model"].unique().tolist())

        # unique grades as numeric strings, sorted by numeric value:
        # e.g. ['3.5', '4.0', '4.5', '5.0']
        grade_options = sorted(
            df["ovrl_grade"].unique().tolist(), key=lambda x: float(x)
        )
    except Exception:
        # Fallback if DB is empty / training df fails
        model_options = ["E400", "E43", "E53_PRE", "E53_FL"]
        grade_options = ["3.5", "4.0", "4.5", "5.0"]

    return render(
        request,
        "price_dashboard.html",
        {
            "model_options": model_options,
            "grade_options": grade_options,
        },
    )


@require_http_methods(["GET"])
def api_history(request):
    """
    Return JSON of historical auction data for the scatter plot.
    Uses the same data pipeline as the ML training (mileage, price, Model, grade).
    Optional ?model=E400 / ?model=E53_PRE / ?model=E53_FL filter.
    """
    try:
        df = _fetch_training_df()  # mileage, ovrl_grade, Model, final_price
    except Exception as e:
        return HttpResponseBadRequest(str(e))

    model = request.GET.get("model")
    if model:
        model = model.upper()
        df = df[df["Model"] == model]

    # sort by mileage for nicer plotting
    df = df.sort_values("mileage")

    points = []
    for _, row in df.iterrows():
        points.append(
            {
                "mileage": int(row["mileage"]),
                "price": float(row["final_price"]),
                "model": row["Model"],        # e.g. 'E53_PRE', 'E53_FL'
                "grade": row["ovrl_grade"],   # e.g. '4.5'
            }
        )

    return JsonResponse({"points": points})
