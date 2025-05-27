import pandas as pd

def load_kaggle_phishbuster_data(file_path="../data/phishing_site_urls.csv"):
    df = pd.read_csv(file_path)
    print("Sample data:\n", df.head())
    print("\nLabel distribution:\n", df['label'].value_counts())
    return df

def save_cleaned_data(df, output_file="../data/cleaned_phishbuster.csv"):
    df.to_csv(output_file, index=False)
    print(f"\nSaved cleaned data to {output_file}")

if __name__ == "__main__":
    df = load_kaggle_phishbuster_data()
    save_cleaned_data(df)
