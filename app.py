import os
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================
# 1. KONFIGURASI HALAMAN WEB
# ==========================================
st.set_page_config(
    page_title="Customer Clustering Dashboard",
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

# ==========================================
# 2. LOAD MODEL & DATASET (DENGAN CACHE)
# ==========================================
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "kmeans_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("File kmeans_model.pkl atau scaler.pkl tidak ditemukan.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "Customers.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError("File Customers.csv tidak ditemukan.")
    return pd.read_csv(data_path)

# Panggil Model & Data
try:
    with st.spinner("Memuat model machine learning dan data..."):
        model, scaler = load_model()
        df = load_data()
except Exception as e:
    st.error(f"❌ Terjadi kesalahan saat memuat file: {e}")
    st.stop()

# Tambahkan prediksi cluster ke dataframe jika belum ada
if "Cluster" not in df.columns:
    try:
        scaled_df = scaler.transform(df[FEATURES])
        df["Cluster"] = model.predict(scaled_df)
    except Exception:
        pass

# ==========================================
# 3. HEADER & METRIK SUMMARY
# ==========================================
st.title("📊 Customer Segmentation Dashboard")
st.markdown("Aplikasi Analisis & Segmentasi Pelanggan Berbasis **K-Means Clustering (CRISP-DM)**")

# Ringkasan Statistik Utama
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pelanggan", f"{len(df):,}")
col2.metric("Rata-rata Usia", f"{df['Age'].mean():.1f} Tahun")
col3.metric("Rata-rata Pendapatan", f"${df['Annual Income ($)'].mean():,.0f}")
col4.metric("Total Cluster", f"{len(np.unique(df['Cluster'])) if 'Cluster' in df.columns else model.n_clusters}")

st.markdown("---")

# ==========================================
# 4. TAB NAVIGASI DASHBOARD
# ==========================================
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Cluster Baru", "📈 Visualisasi Cluster", "📋 Dataset & Summary"])

# ------------------------------------------
# TAB 1: FORM PREDIKSI PELANGGAN BARU
# ------------------------------------------
with tab1:
    st.subheader("📝 Input Data Pelanggan Baru")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        input_data = {}
        for feature in FEATURES:
            if feature in df.columns:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                input_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val
                )
            else:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)

        btn_predict = st.button("🔮 Prediksi Cluster Pelanggan", type="primary", use_container_width=True)

    with col_result:
        st.subheader("💡 Hasil Segmentasi")
        if btn_predict:
            input_df = pd.DataFrame([input_data])
            scaled_input = scaler.transform(input_df[FEATURES])
            cluster_pred = model.predict(scaled_input)[0]
            
            st.success(f"### 🎉 Pelanggan Masuk ke: **Cluster {cluster_pred}**")
            
            # Tampilkan statistik profil input
            st.markdown("**Ringkasan Profil yang Diinput:**")
            st.json(input_data)
        else:
            st.info("Masukkan nilai pada form di sebelah kiri dan klik **Prediksi Cluster Pelanggan**.")

# ------------------------------------------
# TAB 2: VISUALISASI CLUSTER
# ------------------------------------------
with tab2:
    st.subheader("📈 Analisis & Grafik Segmentasi")
    
    col_fig1, col_fig2 = st.columns(2)
    
    with col_fig1:
        st.markdown("**Annual Income vs Spending Score**")
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(
            df["Annual Income ($)"], 
            df["Spending Score (1-100)"], 
            c=df["Cluster"] if "Cluster" in df.columns else 'teal', 
            cmap='viridis', 
            alpha=0.7
        )
        ax.set_xlabel("Annual Income ($)")
        ax.set_ylabel("Spending Score (1-100)")
        plt.colorbar(scatter, label='Cluster')
        st.pyplot(fig)

    with col_fig2:
        st.markdown("**Distribusi Jumlah Pelanggan per Cluster**")
        if "Cluster" in df.columns:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            cluster_counts = df["Cluster"].value_counts().sort_index()
            ax2.bar(cluster_counts.index.astype(str), cluster_counts.values, color='skyblue')
            ax2.set_xlabel("Cluster")
            ax2.set_ylabel("Jumlah Pelanggan")
            st.pyplot(fig2)

# ------------------------------------------
# TAB 3: DATASET & RINGKASAN
# ------------------------------------------
with tab3:
    st.subheader("📋 Dataset Pelanggan")
    st.dataframe(df, use_container_width=True)
    
    st.subheader("📊 Statistik Deskriptif Dataset")
    st.dataframe(df.describe().T, use_container_width=True)