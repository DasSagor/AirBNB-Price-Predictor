import pandas as pd
import os

DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://data.insideairbnb.com/united-states/ny/new-york-city/2025-11-01/data"
FILES = {
    "listings.csv.gz": f"{BASE_URL}/listings.csv.gz",
    "calendar.csv.gz": f"{BASE_URL}/calendar.csv.gz",
    "reviews.csv.gz": f"{BASE_URL}/reviews.csv.gz",
}

for fname, url in FILES.items():
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"Downloading {fname}...")
        try:
            df = pd.read_csv(url)
            df.to_csv(fpath, index=False)
            print(f"✅ {fname} saved ({df.shape[0]} rows)")
        except Exception as e:
            print(f"❌ Failed: {e}")
    else:
        print(f"{fname} already exists.")