"""
intrusion_detection_app.py

This script implements an intrusion detection system using the NSL-KDD dataset.
It supports two modes:

1. Training Mode:
   - Loads the NSL-KDD training dataset from Hugging Face.
   - Performs feature engineering and scaling.
   - Trains a Random Forest classifier and an Isolation Forest.
   - Evaluates performance and saves the models and scaler.

   Usage: python intrusion_detection_app.py train

2. Dashboard Mode:
   - Loads the saved models and scaler.
   - Loads the NSL-KDD test dataset from Hugging Face.
   - Applies the same feature engineering and scaling.
   - Simulates real-time predictions by processing the data row-by-row.
   - Displays live-updating graphs of predictions and anomaly scores using Streamlit.

   Usage: streamlit run intrusion_detection_app.py dashboard
"""

import sys
import time
import warnings
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# For dashboard display with Streamlit
import streamlit as st

# Set environment and warnings as in your notebook
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_columns', None)


##############################
#   Feature Engineering      #
##############################

def feat_gen(data):
    """
    Feature Engineering function:
      - Creates new features based on traffic statistics, attack patterns, and user behavior.
      - Encodes categorical features.
      - Converts output class ('normal' vs. attack) to binary (0 for normal, 1 for anomaly).
    """
    # 1. Traffic Volume & Connection Statistics
    data['avg_duration_per_host'] = data.groupby('dst_host_count')['duration'].transform('mean')
    data['conn_rate'] = data['count'] / (data['duration'] + 1)
    data['bytes_ratio'] = data['src_bytes'] / (data['dst_bytes'] + 1)
    
    # 2. Attack pattern Indicators
    data['failed_login_rate'] = data['num_failed_logins'] / (data['count'] + 1)
    data['error_ratio'] = data['serror_rate'] / (data['srv_serror_rate'] + 1)
    data['rerror_ratio'] = data['rerror_rate'] / (data['srv_rerror_rate'] + 1)
    
    # 3. User behavior
    data['is_guest_access'] = data['is_guest_login'].apply(lambda x: 1 if x == 1 else 0)
    data['shell_access_rate'] = data['num_shells'] / (data['num_access_files'] + 1)
    data['privilege_abuse_score'] = data['num_file_creations'] / (data['num_root'] + 1)
    
    # 4. Categorical Feature Encoding
    # Internet Control Message Protocol, is a network protocol that sends error messages and 
    # other operational information between network devices. 
    protocol_order = {'icmp': 0, 'udp': 1, 'tcp': 2}
    data['protocol_type'] = data['protocol_type'].map(protocol_order)
    
    service_counts = data['service'].value_counts(normalize=True)
    data['service_encoded'] = data['service'].map(service_counts)
    
    flag_order = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'SH': 5, 
                  'S1': 6, 'RSTOS0': 7, 'S3': 8, 'S2': 9, 'OTH': 10}
    data['flag'] = data['flag'].map(flag_order)
    
    # 5. Output Class
    # Print original counts if desired
    print("Before mapping, class counts:\n", data['class'].value_counts())
    data['class'] = data['class'].apply(lambda x: 0 if x == 'normal' else 1)
    print("After mapping, class counts:\n", data['class'].value_counts())
    
    return data


##############################
#      Training Mode         #
##############################

def train_models():
    # Load the NSL-KDD dataset from Hugging Face
    dataset = load_dataset("Mireu-Lab/NSL-KDD")
    train_data = dataset['train'].to_pandas()
    
    # Feature engineering on training data
    train_data = feat_gen(train_data)
    
    # Define columns to be used for supervised training
    # (exclude the original 'class' column)
    supervised_cols = ['avg_duration_per_host', 'conn_rate', 'bytes_ratio', 
                         'failed_login_rate', 'error_ratio', 'rerror_ratio', 
                         'is_guest_access', 'shell_access_rate', 'privilege_abuse_score',
                         'protocol_type', 'service_encoded', 'flag']
    
    # Define numerical features for scaling
    numerical_features = ['avg_duration_per_host', 'conn_rate', 'bytes_ratio', 
                          'failed_login_rate', 'error_ratio', 'rerror_ratio', 
                          'is_guest_access', 'shell_access_rate', 'privilege_abuse_score']
    
    # Fit scaler on training numerical features and transform
    scaler = StandardScaler()
    train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
    
    # Save the scaler for use in the dashboard
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved to disk as 'scaler.pkl'")
    
    # Prepare supervised training data
    X = train_data[supervised_cols]
    y = train_data['class'].astype(int)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ----- Random Forest Training -----
    rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_val = rf_model.predict(X_val)
    print("Random Forest Validation Accuracy:", accuracy_score(y_val, y_pred_val))
    print("Random Forest Classification Report:\n", classification_report(y_val, y_pred_val))
    
    # Save the trained Random Forest model
    joblib.dump(rf_model, 'rf_model.pkl')
    print("Random Forest model saved as 'rf_model.pkl'")
    
    # ----- Isolation Forest Training (Unsupervised) -----
    # For unsupervised training, we choose a subset of features.
    # Here we use features that had higher importance in prior experiments.
    unsupervised_features = ['flag', 'conn_rate', 'service_encoded', 'error_ratio',
                             'protocol_type', 'bytes_ratio', 'rerror_ratio', 'avg_duration_per_host']
    
    unsupervised_data = train_data[unsupervised_features]
    
    iso_forest = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
    # The fit_predict returns -1 for anomalies and 1 for normal instances.
    # We store the raw prediction in a new column.
    unsupervised_data['anomaly_score'] = iso_forest.fit_predict(unsupervised_data)
    # Convert predictions: 1 (normal) -> 0, -1 (anomaly) -> 1.
    unsupervised_data['anomaly'] = (unsupervised_data['anomaly_score'] == -1).astype(int)
    
    # (Optional) Evaluate unsupervised performance using the original labels
    print("Unsupervised model (Isolation Forest) fitted on selected features.")
    
    # Save the Isolation Forest model
    joblib.dump(iso_forest, 'iso_forest_model.pkl')
    print("Isolation Forest model saved as 'iso_forest_model.pkl'")


##############################
#      Dashboard Mode        #
##############################

def run_dashboard():
    st.title("Intrusion Detection Real-Time Dashboard")
    st.write("This dashboard loads the NSL-KDD test dataset, applies the pre-trained models (Random Forest and Isolation Forest), and displays live-updating graphs including overall counts of normal vs intrusions per second.")

    # Load saved models and scaler
    try:
        rf_model = joblib.load('rf_model.pkl')
        iso_forest = joblib.load('iso_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except Exception as e:
        st.error("Error loading models or scaler. Have you run the training mode? " + str(e))
        return

    # Load NSL-KDD test dataset from Hugging Face
    try:
        dataset = load_dataset("Mireu-Lab/NSL-KDD")
        test_data = dataset['test'].to_pandas()
    except Exception as e:
        st.error("Error loading dataset from Hugging Face: " + str(e))
        return

    # Feature engineering on test data
    test_data = feat_gen(test_data)

    # Use the same columns as in training
    supervised_cols = [
        'avg_duration_per_host', 'conn_rate', 'bytes_ratio',
        'failed_login_rate', 'error_ratio', 'rerror_ratio',
        'is_guest_access', 'shell_access_rate', 'privilege_abuse_score',
        'protocol_type', 'service_encoded', 'flag'
    ]

    numerical_features = [
        'avg_duration_per_host', 'conn_rate', 'bytes_ratio',
        'failed_login_rate', 'error_ratio', 'rerror_ratio',
        'is_guest_access', 'shell_access_rate', 'privilege_abuse_score'
    ]

    # Scale test data using the saved scaler
    test_data[numerical_features] = scaler.transform(test_data[numerical_features])

    # For unsupervised predictions, select features that match training
    unsupervised_features = [
        'flag', 'conn_rate', 'service_encoded', 'error_ratio',
        'protocol_type', 'bytes_ratio', 'rerror_ratio', 'avg_duration_per_host'
    ]

    # Display a preview of the dataset
    st.subheader("Test Dataset Preview")
    st.write(test_data.head())

    # -------------------------------------------------------
    # Placeholders for total normal/intrusions at the top
    # -------------------------------------------------------
    col1, col2 = st.columns(2)
    normal_placeholder = col1.empty()      # will hold "Total Normal" metric
    intrusion_placeholder = col2.empty()   # will hold "Total Intrusions" metric

    # Initialize counters
    normal_total = 0
    intrusion_total = 0

    # -------------------------------------------------------
    # Per-Second Chart
    # -------------------------------------------------------
    st.subheader("Per-Second Counts (Normal vs Intrusions)")
    overall_chart = st.line_chart(
        pd.DataFrame({'Time (s)': [], 'Normal': [], 'Intrusions': []}).set_index("Time (s)")
    )

    # -------------------------------------------------------
    # Other Live Charts
    # -------------------------------------------------------
    st.subheader("Random Forest Predictions Over Time")
    rf_chart = st.line_chart(pd.DataFrame({'Prediction': []}))

    st.subheader("Isolation Forest Anomaly Scores Over Time")
    iso_chart = st.line_chart(pd.DataFrame({'Anomaly Score': []}))

    # -------------------------------------------------------
    # Prepare for real-time simulation
    # -------------------------------------------------------
    rf_predictions = []
    iso_scores = []

    current_second = None
    batch_normal = 0
    batch_intrusion = 0
    per_second_counts = []

    # Start simulation timer
    sim_start = time.time()
    delay = 0.5  # seconds delay between processing samples

    # Placeholder for a single status line
    status_placeholder = st.empty()

    st.info("Starting real-time simulation. Predictions and anomaly scores will update as new samples are processed.")

    # -------------------------------------------------------
    # Simulate real-time processing
    # -------------------------------------------------------
    for idx, row in test_data.iterrows():
        # Determine elapsed seconds (integer)
        elapsed = time.time() - sim_start
        sec = int(elapsed)
        if current_second is None:
            current_second = sec

        # If we moved to a new second, update the per-second chart and reset batch counters
        if sec != current_second:
            per_second_counts.append({
                'Time (s)': current_second,
                'Normal': batch_normal,
                'Intrusions': batch_intrusion
            })
            # Update overall chart with the latest second's batch
            overall_chart.add_rows(
                pd.DataFrame([per_second_counts[-1]]).set_index("Time (s)")
            )
            # Reset batch counters and update current second marker
            current_second = sec
            batch_normal = 0
            batch_intrusion = 0

        # --- Random Forest Prediction ---
        sample_rf = row[supervised_cols].values.reshape(1, -1)
        pred = int(rf_model.predict(sample_rf)[0])
        rf_predictions.append(pred)

        # Update overall cumulative metrics and per-second batch counts
        if pred == 0:
            normal_total += 1
            batch_normal += 1
        else:
            intrusion_total += 1
            batch_intrusion += 1

        # Update the metric placeholders (re-using the same widgets each time)
        normal_placeholder.metric("Total Normal", normal_total)
        intrusion_placeholder.metric("Total Intrusions", intrusion_total)

        # --- Isolation Forest Prediction ---
        sample_unsup = row[unsupervised_features].values.reshape(1, -1)
        score = float(iso_forest.decision_function(sample_unsup)[0])
        iso_scores.append(score)

        # Update line charts for the single new sample
        rf_chart.add_rows(pd.DataFrame({'Prediction': [rf_predictions[-1]]}))
        iso_chart.add_rows(pd.DataFrame({'Anomaly Score': [iso_scores[-1]]}))

        # Update status text for the current sample
        status_placeholder.write(
            f"Processed sample {idx+1}/{len(test_data)} at ~{sec} seconds"
        )

        time.sleep(delay)

    # After the loop, update the chart with the final batch if needed
    if batch_normal > 0 or batch_intrusion > 0:
        per_second_counts.append({
            'Time (s)': current_second,
            'Normal': batch_normal,
            'Intrusions': batch_intrusion
        })
        overall_chart.add_rows(
            pd.DataFrame([per_second_counts[-1]]).set_index("Time (s)")
        )

    st.success("Real-time simulation completed.")

##############################
#         Main App           #
##############################

if __name__ == '__main__':
    # Usage:
    #   Training mode: python intrusion_detection_app.py train
    #   Dashboard mode: streamlit run intrusion_detection_app.py dashboard
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        print("Starting training mode...")
        train_models()
    else:
        # When running with Streamlit, sys.argv[1] may be 'dashboard'
        run_dashboard()
