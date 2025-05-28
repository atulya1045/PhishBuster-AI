from utils.predict import predict_url
from utils.db_utils import log_prediction_to_db

# Example URL to test
url = "http://phishy-example.com"

# Predict
result = predict_url(url)
print("🔍 Prediction Result:", result)

# Log to DB
log_prediction_to_db(
    url=result["url"],
    email=None,
    prediction=result["prediction"]
)

print("✅ Prediction logged to DB!")
