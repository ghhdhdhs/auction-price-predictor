import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pricepred.settings")
django.setup()

from pricepred.ml_basic import train_basic

if __name__ == "__main__":
    info = train_basic()
    print(f"Basic model trained on {info['trained_on']} rows with {info['features']} features.")
