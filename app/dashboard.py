# app/dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.db_utils import fetch_all_logs


def run():
    # Fetch data
    df = fetch_all_logs()

    if df.empty:
        st.warning("No data available to display.")
        return

    # Date filter
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        date_range = st.date_input("Filter by date", [])
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['timestamp'] >= pd.to_datetime(start_date)) & 
                    (df['timestamp'] <= pd.to_datetime(end_date))]

    st.subheader("🔍 Raw Detection Logs")
    st.dataframe(df)

    # Phishing vs Legitimate counts
    if 'prediction' in df.columns:
        st.subheader("📊 Detection Summary")
        count_fig = px.histogram(df, x="prediction", color="prediction",
                                 title="Phishing vs Legitimate Detections",
                                 text_auto=True)
        st.plotly_chart(count_fig, use_container_width=True)

    # Most common URLs or spammy senders
    if 'url' in df.columns:
        st.subheader("🌐 Top Detected URLs")
        top_urls = df['url'].value_counts().nlargest(10)
        st.bar_chart(top_urls)

    if 'email' in df.columns:
        st.subheader("📧 Common Email Content Snippets")
        st.write(df['email'].dropna().head(5))
        st.write("Preview of email content snippets that triggered detections.")
    # Feature importance (if available)