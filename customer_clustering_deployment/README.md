# Customer Clustering - Streamlit Deployment

Project clustering pelanggan menggunakan K-Means berdasarkan tahapan CRISP-DM.

## File
- `app.py` : aplikasi Streamlit
- `Customers.csv` : dataset
- `kmeans_model.pkl` : model K-Means terlatih
- `scaler.pkl` : StandardScaler terlatih
- `requirements.txt` : dependency aplikasi

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Upload seluruh file ke repository GitHub, lalu pilih `app.py` sebagai Main file
pada Streamlit Community Cloud.
