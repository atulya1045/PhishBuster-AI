import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocessing import preprocess_data

def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_phishbuster.csv")
    df = pd.read_csv(file_path)
    return df

def split_data(df, label_column):
    X = df.drop(label_column, axis=1)
    y = df[label_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def tune_hyperparameters(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [1, 1.5, 2],
        'min_child_weight': [1, 3, 5]
    }
    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    search = RandomizedSearchCV(model, param_distributions=param_dist,
                                n_iter=30, cv=3, scoring='accuracy',
                                verbose=1, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    print("Best hyperparameters:", search.best_params_)
    return search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("XGBoost Evaluation Report:\n", classification_report(y_test, y_pred))
    print("Test Accuracy:", round(accuracy_score(y_test, y_pred), 4))

def save_model(model):
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    print("Loading data and preprocessing...")
    df = load_data()
    df, label_column = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df, label_column)

    print("Starting hyperparameter tuning...")
    model = tune_hyperparameters(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("Saving model...")
    save_model(model)
    print("Training complete.")
