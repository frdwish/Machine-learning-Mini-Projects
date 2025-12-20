import os
import json
import pickle
import pandas as pd
import numpy as np

from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')   # IMPORTANT for Flask
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open('kmeans_model.pkl', 'rb'))

# -----------------------------
# Load & Clean Data
# -----------------------------
def load_and_clean_data(file_path):
    # Load dataset
    retail = pd.read_csv(
        file_path,
        sep=",",
        encoding="ISO-8859-1",
        header=0
    )

    # Convert CustomerID to string
    retail['CustomerID'] = retail['CustomerID'].astype(str)

    # Create Amount column
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']

    # -----------------------------
    # RFM Metrics
    # -----------------------------
    # Monetary
    rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()

    # Frequency
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']

    # Recency
    retail['InvoiceDate'] = pd.to_datetime(
        retail['InvoiceDate'], format='%d-%m-%Y %H:%M'
    )
    max_date = retail['InvoiceDate'].max()
    retail['Diff'] = max_date - retail['InvoiceDate']

    rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days

    # Merge RFM
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

    # -----------------------------
    # Remove Outliers (IQR)
    # -----------------------------
    Q1 = rfm.quantile(0.05)
    Q3 = rfm.quantile(0.95)
    IQR = Q3 - Q1

    rfm = rfm[
        (rfm.Amount >= Q1[0] - 1.5 * IQR[0]) &
        (rfm.Amount <= Q3[0] + 1.5 * IQR[0]) &
        (rfm.Recency >= Q1[2] - 1.5 * IQR[2]) &
        (rfm.Recency <= Q3[2] + 1.5 * IQR[2]) &
        (rfm.Frequency >= Q1[1] - 1.5 * IQR[1]) &
        (rfm.Frequency <= Q3[1] + 1.5 * IQR[1])
    ]

    return rfm

# -----------------------------
# Preprocess Data
# -----------------------------
def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)

    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]

    scaler = StandardScaler()
    rfm_df_scaled = scaler.fit_transform(rfm_df)

    rfm_df_scaled = pd.DataFrame(
        rfm_df_scaled,
        columns=['Amount', 'Frequency', 'Recency']
    )

    return rfm, rfm_df_scaled

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)

    # Preprocess
    rfm, rfm_scaled = preprocess_data(file_path)

    # Predict clusters
    clusters = model.predict(rfm_scaled)

    rfm['Cluster_Id'] = clusters

    # -----------------------------
    # Generate Plots
    # -----------------------------
    os.makedirs('static', exist_ok=True)

    # Amount vs Cluster
    sns.stripplot(
        x='Cluster_Id',
        y='Amount',
        data=rfm,
        hue='Cluster_Id'
    )
    amount_img = 'static/Cluster_Amount.png'
    plt.savefig(amount_img)
    plt.clf()

    # Frequency vs Cluster
    sns.stripplot(
        x='Cluster_Id',
        y='Frequency',
        data=rfm,
        hue='Cluster_Id'
    )
    freq_img = 'static/Cluster_Frequency.png'
    plt.savefig(freq_img)
    plt.clf()

    # Recency vs Cluster
    sns.stripplot(
        x='Cluster_Id',
        y='Recency',
        data=rfm,
        hue='Cluster_Id'
    )
    rec_img = 'static/Cluster_Recency.png'
    plt.savefig(rec_img)
    plt.clf()

    return render_template(
        'result.html',
        amount_img=amount_img,
        freq_img=freq_img,
        rec_img=rec_img
    )

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
