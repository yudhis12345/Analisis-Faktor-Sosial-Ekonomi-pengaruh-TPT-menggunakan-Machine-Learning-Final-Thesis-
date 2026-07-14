# Analisis-Faktor-Sosial-Ekonomi-pengaruh-TPT-menggunakan-Machine-Learning-Final-Thesis-
Menganalisa beberapa faktor sosial yang diasumsikan sebagai pengaruh terjadinya Tingkat Pengangguran Terbuka yang tinggi di Jawa Tengah dengan mencari faktor dominan nya dan juga membandingkan ketiga metode Machine Learning dan juga Regresi Linear untuk mencari performa prediksi terbaik

# Analisis Faktor Sosial Ekonomi terhadap Tingkat Pengangguran Terbuka di Jawa Tengah

Proyek tugas akhir yang bertujuan untuk menganalisis faktor-faktor sosial ekonomi yang memengaruhi **Tingkat Pengangguran Terbuka (TPT)** pada kabupaten/kota di Provinsi Jawa Tengah menggunakan pendekatan Machine Learning.

Proyek ini membandingkan beberapa metode prediksi, yaitu **Regresi Linear, Random Forest, dan XGBoost**, serta menerapkan optimasi hyperparameter untuk meningkatkan performa model.

## Tujuan Proyek

- Menganalisis faktor sosial ekonomi yang berpengaruh terhadap Tingkat Pengangguran Terbuka (TPT).
- Mengidentifikasi variabel yang paling dominan menggunakan Feature Importance.
- Membandingkan performa Regresi Linear, Random Forest, dan XGBoost.
- Melakukan optimasi hyperparameter pada model Machine Learning.
- Melakukan prediksi TPT pada tingkat kabupaten/kota di Jawa Tengah.
- Menyajikan hasil analisis melalui dashboard interaktif berbasis Streamlit.

## Metode yang Digunakan

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- RandomizedSearchCV
- K-Fold Cross Validation
- Feature Importance

## Evaluasi Model

Performa model dievaluasi menggunakan beberapa metrik:

- R² (Coefficient of Determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

## Teknologi dan Library

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Seaborn
- Statsmodels
- Jupyter Notebook

## Struktur Proyek

- `Persiapan.ipynb` — proses persiapan dan preprocessing data.
- `Method.ipynb` — proses pemodelan dan evaluasi Machine Learning.
- `app.py` — aplikasi dashboard interaktif menggunakan Streamlit.
- `data_final.csv` — dataset yang digunakan dalam analisis.

## Fitur Dashboard

Dashboard Streamlit menyediakan beberapa fitur utama:

- Eksplorasi dan ringkasan dataset.
- Visualisasi distribusi data dan outlier.
- Pemeriksaan missing value.
- Analisis Variance Inflation Factor (VIF).
- Pengaturan train-test split dan parameter model.
- Optimasi hyperparameter menggunakan RandomizedSearchCV.
- Perbandingan performa Regresi Linear, Random Forest, dan XGBoost.
- Analisis Feature Importance.
- Prediksi TPT per kabupaten/kota.
- Analisis hasil prediksi pada tingkat Provinsi Jawa Tengah.
