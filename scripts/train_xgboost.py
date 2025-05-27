import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import xgboost as xgb
from scipy.stats import uniform, randint

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.preprocessing import URLPreprocessor

def load_data():
    file_path = os.path.join(project_root, "data", "cleaned_phishbuster.csv")
    df = pd.read_csv(file_path)
    return df

def split_data(df, label_column):
    X = df.drop(label_column, axis=1)
    y = df[label_column]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def tune_xgboost_hyperparameters(X_train, y_train):
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    param_dist = {
        'n_estimators': randint(100, 200),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.05, 0.15),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'min_child_weight': randint(1, 5),
        'gamma': uniform(0, 0.3)
    }

    rand_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=20,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    rand_search.fit(X_train, y_train)
    print(f"✅ Best cross-validation accuracy: {rand_search.best_score_:.4f}")
    print(f"✅ Best parameters: {rand_search.best_params_}")
    return rand_search.best_estimator_

def evaluate_with_threshold(clf, X_test, y_test, threshold):
    proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    print(f"\n📊 Classification Report with threshold {threshold}:")
    print(classification_report(y_test, y_pred))
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def save_model(clf, threshold):
    model_path = os.path.join(project_root, "models", "xgboost_model.pkl")
    meta_path = os.path.join(project_root, "models", "xgboost_metadata.json")

    joblib.dump(clf, model_path)
    with open(meta_path, 'w') as f:
        json.dump({'threshold': threshold}, f)

    print(f"📁 Model saved to: {model_path}")
    print(f"📁 Threshold saved to: {meta_path}")

if __name__ == "__main__":
    print("📥 Loading data...")
    df = load_data()

    print("🧹 Preprocessing data...")
    preprocessor = URLPreprocessor()
    df, label_column = preprocessor.preprocess_data(df)

    print("🧪 Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, label_column)

    print("🚀 Tuning hyperparameters (Randomized Search)...")
    best_clf = tune_xgboost_hyperparameters(X_train, y_train)

    print("✅ Evaluating on test set with default threshold 0.5:")
    evaluate_with_threshold(best_clf, X_test, y_test, 0.5)

    print("✅ Evaluating on test set with custom threshold 0.60:")
    custom_threshold = 0.60
    evaluate_with_threshold(best_clf, X_test, y_test, custom_threshold)

    save_model(best_clf, custom_threshold)
    print("🎉 Training complete.")
# This script trains an XGBoost model on the PhishBuster dataset, tunes hyperparameters using Randomized Search,