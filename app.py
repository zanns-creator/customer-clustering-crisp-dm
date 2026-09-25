import os
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Konfigurasi Halaman Web
st.set_page_config(
    page_title="Customer Clustering CRISP-DM",
    page_icon="📊",
    layout="wide"
)

FEATURES = [
    "Age",
    "Annual Income ($)",
    "Spending Score (1-100)",
    "Work Experience",
    "Family Size"
]

# 2. Lokasi Direktori Utama
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 3. Fungsi Cache untuk Memuat Model & Scaler
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "kmeans_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("File kmeans_model.pkl atau scaler.pkl tidak ditemukan.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# 4. Fungsi Cache untuk Memuat Dataset
@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "Customers.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError("File Customers.csv tidak ditemukan.")
    return pd.read_csv(data_path)

# 5. Tampilan Utama Aplikasi Streamlit
st.title("📊 Customer Clustering Application")
st.markdown("Aplikasi Segmentasi Pelanggan Berbasis Machine Learning (K-Means Clustering).")

# Panggil Model & Data dengan Spinner Loading & Error Catching
try:
    with st.spinner("Memuat model machine learning dan dataset..."):
        model, scaler = load_model()
        df = load_data()
    st.success("✅ Model dan Dataset Berhasil Dimuat!")
except Exception as e:
    st.error(f"❌ Gagal memuat komponen aplikasi: {e}")
    st.stop()

# 6. Menampilkan Ringkasan Data (Tampilan UI Minimal)
st.subheader("📌 Overview Dataset Pelanggan")
st.dataframe(df.head(10), use_container_width=True)

# 7. Sidebar Input untuk Prediksi/Clustering Baru
st.sidebar.header("Input Data Pelanggan Baru")
input_data = {}
for feature in FEATURES:
    if feature in df.columns:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        input_data[feature] = st.sidebar.number_input(
            f"Masukkan {feature}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )
    else:
        input_data[feature] = st.sidebar.number_input(f"Masukkan {feature}", value=0.0)

# Tombol Prediksi Cluster
if st.sidebar.button("Prediksi Cluster"):
    input_df = pd.DataFrame([input_data])
    scaled_input = scaler.transform(input_df[FEATURES])
    cluster_pred = model.predict(scaled_input)[0]
    
    st.sidebar.success(f"Pelanggan ini masuk ke dalam **Cluster {cluster_pred}**")