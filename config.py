import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')

TRAINING_DATA_FILE = os.path.join(DATA_DIR, 'emails.csv')  # you’ll add this later
MODEL_FILE = os.path.join(MODEL_DIR, 'phishing_model.pkl')

# Parameters
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
MODEL_NAME = "distilbert-base-uncased"
