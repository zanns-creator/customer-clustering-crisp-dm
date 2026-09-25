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

# =========================================================================
# JANGAN HAPUS KODE DI BAWAH INI (Bagian Tampilan & Logika Web Streamlit)
# =========================================================================

# Generate cluster labels using the same preprocessing/model used in the notebook
X = df[FEATURES].copy()
X_scaled = scaler.transform(X)
df["Cluster"] = model.predict(X_scaled)

st.title("Customer Clustering")
st.write(
    "Aplikasi segmentasi pelanggan menggunakan algoritma K-Means "
    "berdasarkan karakteristik usia, pendapatan, skor pengeluaran, "
    "pengalaman kerja, dan ukuran keluarga."
)

tab1, tab2, tab3 = st.tabs(["Dashboard", "Profil Cluster", "Prediksi Pelanggan"])

with tab1:
    st.subheader("Ringkasan Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Pelanggan", f"{len(df):,}")
    c2.metric("Jumlah Fitur Clustering", len(FEATURES))
    c3.metric("Jumlah Cluster", model.n_clusters)

    st.subheader("Distribusi Pelanggan per Cluster")
    counts = df["Cluster"].value_counts().sort_index()
    st.bar_chart(counts)

    st.subheader("Visualisasi Pendapatan vs Spending Score")
    fig, ax = plt.subplots(figsize=(8, 5))
    for cluster in sorted(df["Cluster"].unique()):
        part = df[df["Cluster"] == cluster]
        ax.scatter(
            part["Annual Income ($)"],
            part["Spending Score (1-100)"],
            label=f"Cluster {cluster}"
        )
    ax.set_xlabel("Annual Income ($)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Customer Clusters")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("Profil Rata-rata Setiap Cluster")
    profile = (
        df.groupby("Cluster")[FEATURES]
        .mean()
        .round(2)
    )
    st.dataframe(profile, use_container_width=True)

    st.subheader("Jumlah Anggota Cluster")
    summary = counts.rename("Jumlah Pelanggan").to_frame()
    summary["Persentase (%)"] = (summary["Jumlah Pelanggan"] / len(df) * 100).round(2)
    st.dataframe(summary, use_container_width=True)

    st.info(
        "Nomor cluster merupakan label hasil K-Means. Label 0–4 tidak menunjukkan "
        "tingkatan baik atau buruk; karakteristiknya dilihat dari nilai rata-rata fitur."
    )

with tab3:
    st.subheader("Prediksi Cluster Pelanggan Baru")
    st.write("Masukkan karakteristik pelanggan untuk mengetahui cluster hasil K-Means.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        income = st.number_input(
            "Annual Income ($)", min_value=0, max_value=1000000, value=100000, step=1000
        )
        spending = st.number_input(
            "Spending Score (1-100)", min_value=1, max_value=100, value=50
        )

    with col2:
        work_experience = st.number_input(
            "Work Experience", min_value=0, max_value=100, value=5
        )
        family_size = st.number_input(
            "Family Size", min_value=1, max_value=20, value=3
        )

    if st.button("Prediksi Cluster", type="primary"):
        new_data = pd.DataFrame([{
            "Age": age,
            "Annual Income ($)": income,
            "Spending Score (1-100)": spending,
            "Work Experience": work_experience,
            "Family Size": family_size
        }])

        new_scaled = scaler.transform(new_data[FEATURES])
        prediction = int(model.predict(new_scaled)[0])

        st.success(f"Pelanggan termasuk **Cluster {prediction}**.")

        cluster_profile = profile.loc[prediction]
        st.write("Karakteristik rata-rata cluster tersebut:")
        st.dataframe(cluster_profile.to_frame("Rata-rata"), use_container_width=True)

st.caption("Project Data Science — CRISP-DM Customer Clustering")
