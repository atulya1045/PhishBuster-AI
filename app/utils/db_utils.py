# app/utils/db_utils.py

import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_all_logs():
    # Debug print to confirm values are loaded
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")

    print("🔍 DB Config Loaded:", db_name, db_user, db_host, db_port)

    # Check if any value is None
    if not all([db_name, db_user, db_pass, db_host, db_port]):
        print("❌ Missing one or more DB environment variables.")
        return pd.DataFrame()

    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_pass,
            host=db_host,
            port=db_port
        )
        df = pd.read_sql("SELECT * FROM detection_logs ORDER BY timestamp DESC", conn)
    except Exception as e:
        print(f"❌ DB fetch error: {e}")
        df = pd.DataFrame()
    finally:
        if 'conn' in locals():
            conn.close()

    return df

def log_prediction_to_db(url=None, email=None, prediction="Unknown"):
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO detection_logs (timestamp, url, email, prediction)
            VALUES (NOW(), %s, %s, %s)
        """, (url, email, prediction))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error logging prediction: {e}")
