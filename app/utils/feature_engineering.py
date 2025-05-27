# app/utils/feature_engineering.py

import re
import pandas as pd

def extract_email_features(text: str):
    features = {
        "length": len(text),
        "num_links": len(re.findall(r"http[s]?://", text)),
        "num_exclamations": text.count("!"),
        "has_spammy_words": int(bool(re.search(r"urgent|click here|winner|free|login", text, re.IGNORECASE))),
        "num_uppercase": sum(1 for c in text if c.isupper()),
        "num_words": len(text.split()),
        "suspicious_rate": text.lower().count("suspicious") / (len(text.split()) + 1),
        "num_html_tags": len(re.findall(r"<[^>]+>", text)),
        "num_hex_chars": len(re.findall(r"%[0-9a-fA-F]{2}", text)),
    }
    return pd.DataFrame([features])
