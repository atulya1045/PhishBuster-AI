import pandas as pd
import re

def extract_features(url):
    return {
        "url_length": len(url),
        "has_https": int(url.startswith("https")),
        "num_dots": url.count('.'),
        "has_ip": int(bool(re.match(r'https?://(\d{1,3}\.){3}\d{1,3}', url))),
        "num_hyphens": url.count('-'),
        "num_slashes": url.count('/'),
        "has_at": int('@' in url),
        "has_subdomain": int(url.count('.') > 2),
        "label": 1  # 1 for phishing
    }

def load_openphish_links(file_path="scripts/openphish_links.txt"):
    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls

def save_openphish_features(urls, output_file="scripts/openphish_features.csv"):
    features = [extract_features(url) for url in urls]
    df = pd.DataFrame(features)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rows to {output_file}")

if __name__ == "__main__":
    urls = load_openphish_links()
    save_openphish_features(urls)
