import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os  # Tambahkan library OS untuk mengatur path file secara relatif

st.set_page_config(
    page_title="Customer Clustering",
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

# Ambil lokasi folder tempat file app.py ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    # Menggabungkan lokasi folder aktif dengan nama file model/scaler
    model_path = os.path.join(BASE_DIR, "kmeans_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

@st.cache_data
def load_data():
    # Menggabungkan lokasi folder aktif dengan nama file dataset CSV
    data_path = os.path.join(BASE_DIR, "Customers.csv")
    return pd.read_csv(data_path)

# Panggil fungsi yang jalurnya sudah diperbaiki
model, scaler = load_model()
df = load_data()
