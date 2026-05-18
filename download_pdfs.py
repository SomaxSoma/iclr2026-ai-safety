import os
import time
import requests
import pandas as pd

df = pd.read_csv('safety_results.csv')
safety = df[df['is_safety'] == True]

print(f"Downloading PDFs for {len(safety)} AI safety papers...")

# Create subdomain folders
for subdomain in safety['subdomain'].unique():
    folder = os.path.join('pdfs', subdomain.replace(' & ', '_').replace(' ', '_'))
    os.makedirs(folder, exist_ok=True)

success = 0
skipped = 0
failed = 0

for _, row in safety.iterrows():
    folder = os.path.join('pdfs', row['subdomain'].replace(' & ', '_').replace(' ', '_'))
    filepath = os.path.join(folder, f"{row['id']}.pdf")

    # Skip if already downloaded
    if os.path.exists(filepath):
        skipped += 1
        continue

    url = f"https://openreview.net/pdf?id={row['id']}"

    backoffs = [10, 30, 60]
    attempt = 0
    while True:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"✓ {row['title'][:60]}", flush=True)
                success += 1
                break
            if response.status_code == 429 and attempt < len(backoffs):
                wait = backoffs[attempt]
                print(f"⏳ 429 on {row['title'][:50]} — backing off {wait}s", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            print(f"✗ Failed ({response.status_code}): {row['title'][:60]}", flush=True)
            failed += 1
            break
        except Exception as e:
            print(f"✗ Error: {row['title'][:60]} — {e}", flush=True)
            failed += 1
            break

    time.sleep(5)

print(f"\nDone! Downloaded: {success}, Skipped: {skipped}, Failed: {failed}")
