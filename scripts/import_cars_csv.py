import os, django, math, sys
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pricepred.settings")
django.setup()

import pandas as pd
from pricepred.models import Maker, Chassis, CarSale

CSV_PATH = BASE_DIR / ("cars.csv")  # change if your file is elsewhere

# Map normalized 'model_key' to (variant label, 6-digit chassis code)
# NOTE:
# - E53 is split into pre-facelift and facelift (2021+)
MODEL_TO_CHASSIS = {
    "e400":    ("E400 4MATIC Wagon", "213271"),
    "e43":     ("E43 AMG Wagon",     "213264"),
    "e53_pre": ("E53 Pre-Facelift",  "213261"),
    "e53_fl":  ("E53 Facelift",      "213261"),
}

# Normalise colour -> your choices
COLOR_MAP = {
    "white": "White",
    "black": "Black",
    "silver": "Silver",
    "gray": "Gray",
    "grey": "Gray",
    "blue": "Blue",
    "navy": "Blue",
    "red": "Red",
}

def norm_color(x: str) -> str:
    if not x or (isinstance(x, float) and math.isnan(x)):
        return "Other"
    v = str(x).strip().lower()
    return COLOR_MAP.get(v, "Other")

def to_str_or_blank(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()

def to_int_or_none(x):
    try:
        if x in (None, "", "nan"):
            return None
        v = int(float(x))
        return v
    except Exception:
        return None

def parse_auct_date(s: str):
    # CSV sample shows '23/05/2025'
    return datetime.strptime(s.strip(), "%d/%m/%Y").date()

def run():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Basic column presence check (soft check, we don't hard-fail)
    required = {"model","year","colour","ovrl_grade","int_grade","mileage","final_price","auc_date"}
    missing = required - set(map(str.lower, df.columns))
    if missing:
        print(f"⚠ Warning: CSV is missing expected columns: {missing}")

    # Make sure the maker exists
    maker, _ = Maker.objects.get_or_create(name="Mercedes-Benz")

    created_chassis = 0
    created_sales = 0
    updated_sales = 0
    skipped_missing_price = 0
    skipped_unknown_model = 0

    for _, r in df.iterrows():
        # --- year (we need this BEFORE deciding facelift vs pre-facelift) ---
        year = to_int_or_none(r.get("year"))
        if year is None:
            # If year is missing/bad, we skip (you can change to default 0 if you prefer)
            # print("Skipping row with invalid year:", r)
            continue

        # --- model key from CSV ---
        # e.g. "E400", "E43", "E53"
        model_raw = str(r.get("model", "")).strip().lower()

        # Auto-split E53 into pre-facelift vs facelift based on year
        if model_raw == "e53":
            if year >= 2021:
                model_key = "e53_fl"
            else:
                model_key = "e53_pre"
        else:
            model_key = model_raw

        if model_key not in MODEL_TO_CHASSIS:
            # skip rows we don't map yet (unknown model)
            skipped_unknown_model += 1
            continue

        variant, code = MODEL_TO_CHASSIS[model_key]

        # --- Final price (must be present & >0, otherwise skip row) ---
        final_price = to_int_or_none(r.get("final_price"))
        if final_price is None or final_price <= 0:
            skipped_missing_price += 1
            continue

        # chassis (maker + code + variant)
        ch, ch_created = Chassis.objects.get_or_create(
            maker=maker,
            code=str(code),
            defaults={"variant": variant}
        )
        # If variant label changed, keep it in sync
        if not ch_created and ch.variant != variant:
            ch.variant = variant
            ch.save(update_fields=["variant"])

        created_chassis += int(ch_created)

        # Build CarSale fields
        auction_date   = parse_auct_date(str(r["auc_date"]))
        mileage        = to_int_or_none(r.get("mileage")) or 0
        color          = norm_color(r.get("colour"))
        auction_grade  = to_str_or_blank(r.get("ovrl_grade"))   # e.g. "3.5","4","4.5","5"
        interior_grade = to_str_or_blank(r.get("int_grade"))    # e.g. "A","B","C"
        exterior_grade = to_str_or_blank(r.get("ext_grade"))    # optional
        start_price    = to_int_or_none(r.get("start_price"))   # optional
        auction_house  = ""  # CSV doesn’t have it

        # Upsert logic: use (auction_date + chassis + year + mileage) as a crude natural key
        obj, created = CarSale.objects.update_or_create(
            auction_date=auction_date,
            maker=maker,
            chassis_code=ch,
            year=year,
            mileage=mileage,
            defaults=dict(
                color=color,
                auction_grade=auction_grade,
                interior_grade=interior_grade,
                exterior_grade=exterior_grade,
                start_price=start_price,
                final_price=final_price,
                auction_house=auction_house,
            )
        )
        if created:
            created_sales += 1
        else:
            updated_sales += 1

    print("✅ Import complete.")
    print(f"   Chassis created: {created_chassis}")
    print(f"   CarSale created: {created_sales}, updated: {updated_sales}")
    print(f"   Skipped (unknown model): {skipped_unknown_model}")
    print(f"   Skipped (missing/zero final_price): {skipped_missing_price}")

if __name__ == "__main__":
    run()
