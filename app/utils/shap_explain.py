import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_email_prediction(email_text, model):
    # Vectorize using the same preprocessing inside model
    if hasattr(model, "named_steps"):  # pipeline case
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]
        X_transformed = vectorizer.transform([email_text])
    else:
        classifier = model
        X_transformed = [email_text]  # Assume it handles text input directly

    explainer = shap.Explainer(classifier.predict_proba, X_transformed)
    shap_values = explainer(X_transformed)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.bar(shap_values[0], show=False)
    return fig
    # Note: This assumes the model is compatible with SHAP and can handle text input.