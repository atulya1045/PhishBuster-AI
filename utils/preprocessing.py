# utils/preprocessing/url_preprocessor.py
import pandas as pd
from typing import Tuple

class URLPreprocessor:
    def __init__(self):
        self.required_columns = {'url', 'label'}
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract URL-based features"""
        df['url_length'] = df['url'].apply(lambda x: len(str(x)))
        df['num_dots'] = df['url'].apply(lambda x: str(x).count('.'))
        df['has_at_symbol'] = df['url'].apply(lambda x: '@' in str(x)).astype(int)
        df['has_hyphen'] = df['url'].apply(lambda x: '-' in str(x)).astype(int)
        df['has_https'] = df['url'].apply(lambda x: 'https' in str(x)).astype(int)
        df['has_double_slash'] = df['url'].apply(lambda x: '//' in str(x)).astype(int)
        return df

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Preprocess URL dataset"""
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Validate required columns
        missing_cols = self.required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Clean data
        df = df.dropna(subset=['url', 'label'])
        df = self.extract_features(df)
        
        # Select final features
        feature_cols = [
            'url_length', 'num_dots', 'has_at_symbol', 
            'has_hyphen', 'has_https', 'has_double_slash', 
            'label'
        ]
        return df[feature_cols], 'label'