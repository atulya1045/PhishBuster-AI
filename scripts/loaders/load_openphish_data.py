import csv

def load_openphish_links(file_path="openphish_links.txt"):
    with open(file_path, "r") as file:
        links = [line.strip() for line in file if line.strip()]
    return links

# Use your exact feature extraction function from the UCI script here
def extract_features(url):
    # Example placeholder — replace with your original UCI feature extractor logic
    features = []

    # Sample feature extraction in the style of UCI dataset (change according to your actual logic):
    
    # Feature 1: Is URL length > 75? (1 if yes, else 0)
    features.append(1 if len(url) > 75 else 0)
    
    # Feature 2: Does URL contain '@'? (-1 if yes, else 1)
    features.append(-1 if '@' in url else 1)
    
    # Feature 3: Does URL contain '-'? (1 if yes else -1)
    features.append(1 if '-' in url else -1)
    
    # Feature 4: Count of digits, thresholded (1 if > 3 else 0)
    digit_count = sum(c.isdigit() for c in url)
    features.append(1 if digit_count > 3 else 0)
    
    # Feature 5: HTTPS usage (1 if starts with https, else -1)
    features.append(1 if url.startswith("https") else -1)
    
    # Add as many features as in your UCI phishing extractor, following the same pattern...
    
    return features

def process_openphish_data():
    urls = load_openphish_links()
    feature_list = [extract_features(url) for url in urls]
    return feature_list

def save_features_to_csv(features, output_file="openphish_features.csv"):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Use the same header as UCI dataset CSV
        writer.writerow([
            "length_gt_75", "contains_at", "contains_dash", "digit_count_gt_3", "uses_https"
            # add your other feature names here to match your UCI features
        ])
        for row in features:
            writer.writerow(row)

if __name__ == "__main__":
    features = process_openphish_data()
    save_features_to_csv(features)
    print(f"Processed {len(features)} URLs and saved features to 'openphish_features.csv'")
