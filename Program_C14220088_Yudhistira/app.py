"""
Dashboard Interaktif — Analisis Tingkat Pengangguran Terbuka (TPT)
Provinsi Jawa Tengah
Metode: Regresi Linear, Random Forest, XGBoost (dengan Optimasi Parameter)
Author: Yudhistira — C14220088
[REVISI] Tambahan: Feature Importance Top-3 Kota + Interpolasi TPT Provinsi
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# -----------------------------------------
# KONFIGURASI HALAMAN
# -----------------------------------------
st.set_page_config(
    page_title="Dashboard TPT Jawa Tengah",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1a237e;
        text-align: center; padding: 1rem 0 0.5rem;
    }
    .sub-header {
        font-size: 1rem; color: #555;
        text-align: center; margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.2rem; font-weight: 600; color: #283593;
        border-bottom: 2px solid #c5cae9;
        padding-bottom: 0.3rem; margin: 1.5rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Dashboard Analisis Tingkat Pengangguran Terbuka (TPT)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Provinsi Jawa Tengah · Regresi Linear, Random Forest dan XGBoost dengan Optimasi Parameter</div>', unsafe_allow_html=True)

# -----------------------------------------
# SIDEBAR / MENU SAMPING
# -----------------------------------------
st.sidebar.markdown("## Menu Pengaturan")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Unggah Dataset (.csv)", type=["csv"],
    help="Unggah file CSV data sosial-ekonomi wilayah Jawa Tengah"
)

# [REVISI] Tambahkan uploader untuk file angkatan kerja per tahun
st.sidebar.markdown("---")
st.sidebar.markdown("### [REVISI] Data Angkatan Kerja BPS")
ak_files = st.sidebar.file_uploader(
    "Unggah CSV Angkatan Kerja 2018–2023 (6 file)",
    type=["csv"], accept_multiple_files=True,
    help="Upload 6 file CSV angkatan kerja BPS per tahun (2018–2023)"
)

FEATURE_COLS = [
    "PDRB_ADHK", "IPM", "TPAK",
    "Jumlah_Industri", "Jumlah_Tenaga_Kerja",
    "Distribusi_Penduduk", "Pertumbuhan_Penduduk", "Kepadatan_Penduduk",
]
FEATURE_COLS_V2 = FEATURE_COLS + ["Kota_encoded", "Tahun"]
TARGET_COL = "TPT"

st.sidebar.markdown("---")
st.sidebar.markdown("### Pengaturan Model")
test_size     = st.sidebar.slider("Proporsi Data Pengujian", 0.10, 0.40, 0.20, 0.05)
random_state  = st.sidebar.number_input("Pola Acak (Random State)", value=42, step=1)
n_iter_tuning = st.sidebar.slider("Jumlah Iterasi Pencarian Parameter", 10, 100, 50, 10)
n_estimators  = st.sidebar.select_slider(
    "Jumlah Pohon Keputusan (n_estimators)",
    options=[100, 200, 300, 400, 500], value=300
)

run_button = st.sidebar.button("Jalankan Analisis", use_container_width=True, type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("Mahasiswa:\n**Yudhistira · C14220088**\n\nUniversitas Kristen Petra")

# -----------------------------------------
# FUNGSI PEMBANTU
# -----------------------------------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model(name, y_true, y_pred):
    return {
        "Model"   : name,
        "R²"      : round(r2_score(y_true, y_pred), 4),
        "RMSE"    : round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE"     : round(mean_absolute_error(y_true, y_pred), 4),
        "MAPE (%)": round(mape(y_true, y_pred), 4),
    }

@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    df["Kabupaten/Kota"] = df["Kabupaten/Kota"].str.strip()
    df["Tahun"] = df["Tahun"].astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_ak_files(files):
    """Load & gabungkan semua CSV angkatan kerja menjadi satu DataFrame."""
    ak_list = []
    for f in files:
        tmp = pd.read_csv(f)
        # Cari kolom tahun dari nama file
        fname = f.name
        year = None
        for y in range(2018, 2030):
            if str(y) in fname:
                year = y
                break
        if year is None:
            continue
        col_ak = [c for c in tmp.columns if "Jumlah Angkatan Kerja" in c]
        if not col_ak:
            continue
        tmp = tmp[["Kabupaten/Kota", col_ak[0]]].copy()
        tmp.columns = ["Kabupaten/Kota", "Angkatan_Kerja"]
        tmp["Kabupaten/Kota"] = tmp["Kabupaten/Kota"].astype(str).str.strip()
        exclude = {"", "nan", "Keterangan", "-", "Jawa Tengah"}
        tmp = tmp[~tmp["Kabupaten/Kota"].isin(exclude) & tmp["Angkatan_Kerja"].notna()]
        tmp["Tahun"] = year
        ak_list.append(tmp)
    if not ak_list:
        return None
    return pd.concat(ak_list, ignore_index=True)

def interpolasi_provinsi(df_sub, tpt_col):
    num = (df_sub[tpt_col] * df_sub["Angkatan_Kerja"]).sum()
    den = df_sub["Angkatan_Kerja"].sum()
    return num / den

# -----------------------------------------
# VALIDASI UNGGAH FILE
# -----------------------------------------
if uploaded_file is None:
    st.info("Silakan unggah file CSV di menu sebelah kiri untuk memulai analisis data.")
    st.markdown("""
### Format Kolom Dataset yang Diperlukan:
| Nama Kolom | Keterangan Variabel |
|---|---|
| `Kabupaten/Kota` | Nama Wilayah Kabupaten atau Kota |
| `Tahun` | Tahun Pencatatan Data |
| `TPT` | **Target Utama** — Tingkat Pengangguran Terbuka (%) |
| `PDRB_ADHK` | Produk Domestik Regional Bruto (Atas Dasar Harga Konstan) |
| `IPM` | Indeks Pembangunan Manusia |
| `TPAK` | Tingkat Partisipasi Angkatan Kerja |
| `Jumlah_Industri` | Total Jumlah Industri |
| `Jumlah_Tenaga_Kerja` | Total Jumlah Tenaga Kerja |
| `Distribusi_Penduduk` | Persentase Distribusi Penduduk |
| `Pertumbuhan_Penduduk` | Laju Pertumbuhan Penduduk (%) |
| `Kepadatan_Penduduk` | Tingkat Kepadatan Penduduk |
    """)
    st.stop()

data_raw = load_data(uploaded_file)
missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in data_raw.columns]
if missing_cols:
    st.error(f"Kolom data berikut tidak ditemukan di file Anda: **{missing_cols}**")
    st.stop()

data = data_raw.drop_duplicates().reset_index(drop=True)

# ── Label Encode Kota & merge Angkatan Kerja ─────────────────────────────────
le = LabelEncoder()
data["Kota_encoded"] = le.fit_transform(data["Kabupaten/Kota"])

df_ak = None
if ak_files:
    df_ak = load_ak_files(ak_files)
    if df_ak is not None:
        data = data.merge(df_ak, on=["Kabupaten/Kota", "Tahun"], how="left")

# -----------------------------------------
# MANAJEMEN SESSION STATE
# -----------------------------------------
ak_key = str(len(ak_files)) if ak_files else "0"
state_key = f"{uploaded_file.name}|{len(data)}|{test_size}|{random_state}|{n_iter_tuning}|{n_estimators}|{ak_key}"

if st.session_state.get("state_key") != state_key:
    for k in ["res", "models_trained"]:
        st.session_state.pop(k, None)
    st.session_state["state_key"] = state_key

# -----------------------------------------
# FUNGSI PELATIHAN MODEL
# -----------------------------------------
def run_all(data, feature_cols, feature_cols_v2, target_col,
            test_size, random_state, n_iter, n_estimators):

    # ── [REVISI] Gunakan fitur V2 (dengan Kota_encoded & Tahun) ───────────────
    X = data[feature_cols_v2]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )

    # ── Regresi Linear ────────────────────────────────────────────────────────
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    kf = KFold(n_splits=5, shuffle=True, random_state=int(random_state))

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_param_grid = {
        "n_estimators"     : [n_estimators],
        "max_depth"        : [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
        "max_features"     : ["sqrt", "log2", 0.5],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=int(random_state)),
        rf_param_grid, n_iter=n_iter, cv=kf,
        scoring="neg_mean_squared_error",
        random_state=int(random_state), n_jobs=-1, verbose=0,
    )
    rf_search.fit(X_train, y_train)
    rf_best    = rf_search.best_estimator_
    y_pred_rf  = rf_best.predict(X_test)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_param_grid = {
        "n_estimators"    : [n_estimators],
        "max_depth"       : [3, 5, 7, 9],
        "learning_rate"   : [0.01, 0.05, 0.1, 0.2],
        "subsample"       : [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma"           : [0, 0.1, 0.3],
    }
    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=int(random_state), verbosity=0),
        xgb_param_grid, n_iter=n_iter, cv=kf,
        scoring="neg_mean_squared_error",
        random_state=int(random_state), n_jobs=-1, verbose=0,
    )
    xgb_search.fit(X_train, y_train)
    xgb_best    = xgb_search.best_estimator_
    y_pred_xgb  = xgb_best.predict(X_test)

    # ── Feature Importance — semua kota ──────────────────────────────────────
    fi_rf = pd.DataFrame({
        "Variabel"  : X_train.columns,
        "Importance": rf_best.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    fi_xgb = pd.DataFrame({
        "Variabel"  : X_train.columns,
        "Importance": xgb_best.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # ── [REVISI] Feature Importance — Top-3 kota berpenduduk terbanyak ───────
    top3 = (data.groupby("Kabupaten/Kota")["Distribusi_Penduduk"]
                .mean().sort_values(ascending=False).head(3).index.tolist())

    results_fi_top3 = {}
    for kota in top3:
        sub = data[data["Kabupaten/Kota"] == kota][feature_cols + [target_col]].copy()
        X_k, y_k = sub[feature_cols], sub[target_col]
        rf_k  = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_k, y_k)
        xgb_k = XGBRegressor(n_estimators=300, random_state=42, verbosity=0).fit(X_k, y_k)
        fi_df = pd.DataFrame({
            "Variabel"     : feature_cols,
            "Random Forest": rf_k.feature_importances_,
            "XGBoost"      : xgb_k.feature_importances_,
        })
        fi_df["Rata-rata"] = (fi_df["Random Forest"] + fi_df["XGBoost"]) / 2
        fi_df = fi_df.sort_values("Rata-rata", ascending=False).reset_index(drop=True)
        results_fi_top3[kota] = fi_df

    # ── [REVISI] Prediksi per kota + interpolasi provinsi ────────────────────
    data_pred = data.copy()
    data_pred["TPT_pred_LR"]  = lr_model.predict(data[feature_cols_v2])
    data_pred["TPT_pred_RF"]  = rf_best.predict(data[feature_cols_v2])
    data_pred["TPT_pred_XGB"] = xgb_best.predict(data[feature_cols_v2])

    df_prov = None
    if "Angkatan_Kerja" in data_pred.columns:
        rows = []
        for tahun in sorted(data_pred["Tahun"].unique()):
            sub = data_pred[data_pred["Tahun"] == tahun]
            rows.append({
                "Tahun"                 : tahun,
                "TPT_Aktual_Provinsi"   : interpolasi_provinsi(sub, "TPT"),
                "TPT_Pred_LR_Provinsi"  : interpolasi_provinsi(sub, "TPT_pred_LR"),
                "TPT_Pred_RF_Provinsi"  : interpolasi_provinsi(sub, "TPT_pred_RF"),
                "TPT_Pred_XGB_Provinsi" : interpolasi_provinsi(sub, "TPT_pred_XGB"),
            })
        df_prov = pd.DataFrame(rows)

    # ── Evaluasi ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame([
        evaluate_model("Regresi Linear (Model Dasar)", y_test, y_pred_lr),
        evaluate_model("Random Forest (Optimasi)",     y_test, y_pred_rf),
        evaluate_model("XGBoost (Optimasi)",           y_test, y_pred_xgb),
    ])

    return {
        "X_train"         : X_train, "X_test": X_test,
        "y_train"         : y_train, "y_test": y_test,
        "lr_model"        : lr_model,
        "rf_best"         : rf_best, "xgb_best": xgb_best,
        "y_pred_lr"       : y_pred_lr,
        "y_pred_rf"       : y_pred_rf,
        "y_pred_xgb"      : y_pred_xgb,
        "rf_best_params"  : rf_search.best_params_,
        "xgb_best_params" : xgb_search.best_params_,
        "fi_rf"           : fi_rf, "fi_xgb": fi_xgb,
        "results_df"      : results_df,
        "top3"            : top3,
        "results_fi_top3" : results_fi_top3,
        "data_pred"       : data_pred,
        "df_prov"         : df_prov,
    }

# -----------------------------------------
# TOMBOL PROSES
# -----------------------------------------
res = st.session_state.get("res", None)

if run_button:
    with st.spinner("Mohon tunggu, model sedang menganalisis data..."):
        res = run_all(data, FEATURE_COLS, FEATURE_COLS_V2, TARGET_COL,
                      test_size, int(random_state), n_iter_tuning, n_estimators)
    st.session_state["res"] = res
    st.session_state["models_trained"] = True
    st.success("Analisis selesai! Silakan lihat hasil pada tab di bawah ini.")

# -----------------------------------------
# TAB NAVIGASI HALAMAN
# -----------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data dan Ringkasan Awal",
    "Persiapan Data",
    "Hasil Pengaturan Model",
    "Perbandingan Akurasi Model",
    "Uji Coba Model Default vs Optimasi",
    "[REVISI] Analisis Lanjutan",   # TAB BARU
])

# ==========================================
# TAB 1 — DATA DAN RINGKASAN AWAL
# ==========================================
with tab1:
    st.markdown('<div class="section-title">Ringkasan Ukuran Dataset</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Baris Data", f"{len(data):,}")
    c2.metric("Total Kolom", len(data.columns))
    c3.metric("Data yang Kosong", int(data.isnull().sum().sum()))
    c4.metric("Data Ganda (Dihapus)", int(len(data_raw) - len(data)))

    st.markdown('<div class="section-title">Melihat Isi Data</div>', unsafe_allow_html=True)
    n_show = st.slider("Jumlah baris yang ditampilkan:", 5, 50, 10, key="eda_nrows")
    st.dataframe(data.head(n_show), use_container_width=True)

    st.markdown('<div class="section-title">Ringkasan Statistik Angka Kunci Data</div>', unsafe_allow_html=True)
    st.dataframe(data[FEATURE_COLS + [TARGET_COL]].describe().T.round(4), use_container_width=True)

    st.markdown('<div class="section-title">Grafik Persebaran Angka Variabel</div>', unsafe_allow_html=True)
    cols_to_plot = FEATURE_COLS + [TARGET_COL]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols_to_plot):
        sns.histplot(data[col], kde=True, ax=axes[i], color="#3949ab", edgecolor="white")
        axes[i].set_title(col, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("")
    for j in range(len(cols_to_plot), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("EDA Univariat", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Grafik Deteksi Nilai Ekstrim (Outlier)</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(13, 5))
    data[FEATURE_COLS + [TARGET_COL]].boxplot(ax=ax2)
    ax2.set_xticklabels(FEATURE_COLS + [TARGET_COL], rotation=30, ha="right", fontsize=9)
    ax2.set_title("Boxplot Variabel Penelitian", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2); plt.close()

    if "Tahun" in data.columns:
        st.markdown('<div class="section-title">Grafik Perkembangan TPT dari Tahun ke Tahun</div>', unsafe_allow_html=True)
        tren = data.groupby("Tahun")[TARGET_COL].mean().reset_index()
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(tren["Tahun"], tren[TARGET_COL], marker="o", linewidth=2, color="#1a237e")
        ax4.fill_between(tren["Tahun"], tren[TARGET_COL], alpha=0.15, color="#3949ab")
        ax4.set_title("Rata-rata TPT per Tahun", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Tahun"); ax4.set_ylabel("Rata-rata TPT (%)"); ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4); plt.close()

# ==========================================
# TAB 2 — PERSIAPAN DATA
# ==========================================
with tab2:
    st.markdown('<div class="section-title">Pemeriksaan Data Kosong</div>', unsafe_allow_html=True)
    mv = data[FEATURE_COLS + [TARGET_COL]].isnull().sum().reset_index()
    mv.columns = ["Nama Variabel", "Jumlah Data Kosong"]
    mv["Status Analisis"] = mv["Jumlah Data Kosong"].apply(
        lambda x: "Aman" if x == 0 else "Perlu Diperbaiki"
    )
    st.dataframe(mv, use_container_width=True)

    st.markdown('<div class="section-title">Uji VIF</div>', unsafe_allow_html=True)
    X_vif = data[FEATURE_COLS].dropna()
    vif_df = pd.DataFrame({
        "Nama Variabel": X_vif.columns,
        "Nilai VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
    })
    vif_df["Kondisi"] = vif_df["Nilai VIF"].apply(
        lambda v: "Aman (< 10)" if v < 10 else "Perlu Diperhatikan"
    )
    vif_df = vif_df.sort_values("Nilai VIF", ascending=False)
    st.dataframe(vif_df.round(4), use_container_width=True)

    fig_vif, ax_vif = plt.subplots(figsize=(8, 4))
    c_vif = ["#e53935" if v >= 10 else "#43a047" for v in vif_df["Nilai VIF"]]
    ax_vif.barh(vif_df["Nama Variabel"], vif_df["Nilai VIF"], color=c_vif, edgecolor="black")
    ax_vif.axvline(x=10, color="red", linestyle="--", linewidth=1.5, label="Batas VIF = 10")
    ax_vif.set_title("Grafik Nilai VIF", fontsize=12, fontweight="bold"); ax_vif.legend()
    plt.tight_layout(); st.pyplot(fig_vif); plt.close()

    st.markdown('<div class="section-title">Informasi Pembagian Data</div>', unsafe_allow_html=True)
    n_total = len(data); n_test = int(np.round(n_total * test_size)); n_train = n_total - n_test
    ca, cb, cc = st.columns(3)
    ca.metric("Total Data", n_total)
    cb.metric(f"Data Training ({int((1-test_size)*100)}%)", n_train)
    cc.metric(f"Data Testing ({int(test_size*100)}%)", n_test)
    st.info(f"Split {int((1-test_size)*100)}/{int(test_size*100)} | Random State: {int(random_state)} | Pohon: {n_estimators}")

# ==========================================
# TAB 3 — HASIL PENGATURAN MODEL
# ==========================================
with tab3:
    st.markdown('<div class="section-title">Status Pelatihan Model</div>', unsafe_allow_html=True)
    if res is None:
        st.info("Silakan klik tombol 'Jalankan Analisis' pada menu di sebelah kiri terlebih dahulu.")
        st.stop()

    st.markdown('<div class="section-title">Koefisien Regresi Linear</div>', unsafe_allow_html=True)
    lr = res["lr_model"]
    coef_df = pd.DataFrame({
        "Nama Variabel": res["X_train"].columns, "Nilai Koefisien": lr.coef_,
    }).sort_values("Nilai Koefisien", key=abs, ascending=False)
    st.metric("Intercept", f"{lr.intercept_:.4f}")
    st.dataframe(coef_df.round(4), use_container_width=True)

    col_rf, col_xgb = st.columns(2)
    with col_rf:
        st.markdown('<div class="section-title">Best Params — Random Forest</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(list(res["rf_best_params"].items()), columns=["Parameter","Nilai"]), use_container_width=True)
    with col_xgb:
        st.markdown('<div class="section-title">Best Params — XGBoost</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(list(res["xgb_best_params"].items()), columns=["Parameter","Nilai"]), use_container_width=True)

    st.markdown('<div class="section-title">Feature Importance — Semua Kota</div>', unsafe_allow_html=True)
    fi_rf = res["fi_rf"]; fi_xgb = res["fi_xgb"]
    fig_fi, axes_fi = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=fi_rf, x="Importance", y="Variabel", palette="Blues_r", ax=axes_fi[0], edgecolor="black")
    axes_fi[0].set_title("Feature Importance — Random Forest", fontsize=11, fontweight="bold")
    sns.barplot(data=fi_xgb, x="Importance", y="Variabel", palette="Oranges_r", ax=axes_fi[1], edgecolor="black")
    axes_fi[1].set_title("Feature Importance — XGBoost", fontsize=11, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig_fi); plt.close()

# ==========================================
# TAB 4 — PERBANDINGAN AKURASI MODEL
# ==========================================
with tab4:
    st.markdown('<div class="section-title">Evaluasi Tingkat Akurasi</div>', unsafe_allow_html=True)
    if res is None:
        st.info("Silakan jalankan analisis terlebih dahulu."); st.stop()

    y_test = res["y_test"]; y_pred_lr = res["y_pred_lr"]
    y_pred_rf = res["y_pred_rf"]; y_pred_xgb = res["y_pred_xgb"]
    results_df = res["results_df"]

    best_row = results_df.loc[results_df["MAPE (%)"].idxmin()]
    st.success(f"Model Terbaik: {best_row['Model']} — MAPE={best_row['MAPE (%)']:.4f}% | R²={best_row['R²']:.4f}")

    def highlight_best(col):
        if col.name == "R²":
            return ["background-color: #c8e6c9" if v == col.max() else "" for v in col]
        elif col.name in ["RMSE","MAE","MAPE (%)"]:
            return ["background-color: #c8e6c9" if v == col.min() else "" for v in col]
        return [""] * len(col)

    st.dataframe(results_df.style.apply(highlight_best, axis=0).format(
        {"R²":"{:.4f}","RMSE":"{:.4f}","MAE":"{:.4f}","MAPE (%)":"{:.4f}"}
    ), use_container_width=True)

    st.markdown('<div class="section-title">Visualisasi Perbandingan Error</div>', unsafe_allow_html=True)
    fig_ev, axes_ev = plt.subplots(1, 3, figsize=(15, 5))
    fig_ev.suptitle("Perbandingan Nilai Error Ketiga Model", fontsize=13, fontweight="bold")
    for ax, metric, color in zip(axes_ev, ["RMSE","MAE","MAPE (%)"], ["#2196F3","#4CAF50","#FF9800"]):
        bars = ax.bar(results_df["Model"], results_df[metric], color=color, alpha=0.85, edgecolor="black")
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xticklabels(results_df["Model"], rotation=15, ha="right", fontsize=8)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig_ev); plt.close()

    st.markdown('<div class="section-title">Grafik Aktual vs Prediksi</div>', unsafe_allow_html=True)
    model_preds = [("Regresi Linear", y_pred_lr, "#2196F3"),
                   ("Random Forest",  y_pred_rf, "#4CAF50"),
                   ("XGBoost",        y_pred_xgb, "#FF5722")]
    fig_av, axes_av = plt.subplots(1, 3, figsize=(16, 5))
    fig_av.suptitle("Sebaran Nilai Aktual vs Prediksi", fontsize=13, fontweight="bold")
    for ax, (name, y_pred, color) in zip(axes_av, model_preds):
        ax.scatter(y_test, y_pred, alpha=0.7, color=color, edgecolors="black", linewidth=0.5)
        lim = [min(y_test.min(), y_pred.min())-0.3, max(y_test.max(), y_pred.max())+0.3]
        ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect Fit")
        ax.set_title(f"{name}\nR²={r2_score(y_test, y_pred):.4f}", fontsize=10)
        ax.set_xlabel("Aktual TPT (%)"); ax.set_ylabel("Prediksi TPT (%)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig_av); plt.close()

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("Unduh Tabel Akurasi (.csv)",
            results_df.to_csv(index=False).encode("utf-8"),
            "evaluasi_model_tpt.csv", "text/csv", use_container_width=True)
    with col_dl2:
        pred_compare = pd.DataFrame({
            "Aktual": np.array(y_test),
            "Pred LR": y_pred_lr, "Pred RF": y_pred_rf, "Pred XGB": y_pred_xgb,
        }).round(4)
        st.download_button("Unduh Tabel Prediksi (.csv)",
            pred_compare.to_csv(index=False).encode("utf-8"),
            "prediksi_lengkap.csv", "text/csv", use_container_width=True)

# ==========================================
# TAB 5 — UJI COBA DEFAULT VS OPTIMASI
# ==========================================
with tab5:
    st.markdown('<div class="section-title">Eksperimen: Model Standar vs Optimasi</div>', unsafe_allow_html=True)
    if res is None:
        st.warning("Silakan jalankan analisis utama terlebih dahulu."); st.stop()

    run_variasi = st.button("Mulai Jalankan Komparasi", type="primary", use_container_width=True)
    variasi_key = f"variasi|{state_key}"

    if run_variasi:
        with st.spinner("Menghitung model standar..."):
            X = data[FEATURE_COLS_V2]; y = data[TARGET_COL]
            X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state))
            rf_default  = RandomForestRegressor(random_state=int(random_state)).fit(X_train_v, y_train_v)
            xgb_default = XGBRegressor(random_state=int(random_state), verbosity=0).fit(X_train_v, y_train_v)
            y_pred_rf_default  = rf_default.predict(X_test_v)
            y_pred_xgb_default = xgb_default.predict(X_test_v)
            y_pred_rf_tuned    = res["rf_best"].predict(X_test_v)
            y_pred_xgb_tuned   = res["xgb_best"].predict(X_test_v)
            y_pred_lr_v        = res["lr_model"].predict(X_test_v)

            variasi_rows = [
                evaluate_model("Regresi Linear",               y_test_v, y_pred_lr_v),
                evaluate_model("Random Forest (Standar)",      y_test_v, y_pred_rf_default),
                evaluate_model("Random Forest (Optimasi)",     y_test_v, y_pred_rf_tuned),
                evaluate_model("XGBoost (Standar)",            y_test_v, y_pred_xgb_default),
                evaluate_model("XGBoost (Optimasi)",           y_test_v, y_pred_xgb_tuned),
            ]
            variasi_df = pd.DataFrame(variasi_rows)
            st.session_state["variasi_df"]  = variasi_df
            st.session_state["variasi_key"] = variasi_key
        st.success("Komparasi selesai!")

    if "variasi_df" not in st.session_state:
        st.info("Klik tombol di atas untuk memproses komparasi."); st.stop()

    variasi_df = st.session_state["variasi_df"]

    def highlight_variasi(col):
        if col.name == "R²":
            return ["background-color: #c8e6c9" if v == col.max() else "" for v in col]
        elif col.name in ["RMSE","MAE","MAPE (%)"]:
            return ["background-color: #c8e6c9" if v == col.min() else "" for v in col]
        return [""] * len(col)

    st.dataframe(variasi_df.style.apply(highlight_variasi, axis=0).format(
        {"R²":"{:.4f}","RMSE":"{:.4f}","MAE":"{:.4f}","MAPE (%)":"{:.4f}"}
    ), use_container_width=True)

# ==========================================
# TAB 6 — [REVISI] ANALISIS LANJUTAN
# ==========================================
with tab6:
    st.markdown('<div class="section-title">[REVISI] Feature Importance — 3 Kota Berpenduduk Terbanyak</div>', unsafe_allow_html=True)

    if res is None:
        st.info("Silakan jalankan analisis terlebih dahulu."); st.stop()

    top3            = res["top3"]
    results_fi_top3 = res["results_fi_top3"]

    st.info(f"3 Kabupaten/Kota dengan rata-rata Distribusi_Penduduk tertinggi: **{', '.join(top3)}**")

    # Tabel per kota
    for kota in top3:
        st.markdown(f"**{kota}**")
        st.dataframe(results_fi_top3[kota].round(4), use_container_width=True)

    # Visualisasi
    fig_top3, axes_top3 = plt.subplots(1, 3, figsize=(18, 6))
    for ax, kota in zip(axes_top3, top3):
        fi = results_fi_top3[kota].sort_values("Rata-rata")
        idx = range(len(fi)); w = 0.35
        ax.barh([i - w/2 for i in idx], fi["Random Forest"], height=w,
                label="Random Forest", color="steelblue", alpha=0.8)
        ax.barh([i + w/2 for i in idx], fi["XGBoost"], height=w,
                label="XGBoost", color="darkorange", alpha=0.8)
        ax.set_yticks(list(idx)); ax.set_yticklabels(fi["Variabel"].tolist(), fontsize=9)
        ax.set_title(f"Feature Importance\n{kota}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance Score"); ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    plt.suptitle("Feature Importance — Top 3 Kota Berpenduduk Terbanyak",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(); st.pyplot(fig_top3); plt.close()

    # ── Prediksi per kota ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">[REVISI] Prediksi TPT per Kabupaten/Kota (Tahun 2023)</div>', unsafe_allow_html=True)
    data_pred = res["data_pred"]
    pred_2023 = (data_pred[data_pred["Tahun"] == 2023]
                 [["Kabupaten/Kota", "TPT", "TPT_pred_LR", "TPT_pred_RF", "TPT_pred_XGB"]]
                 .copy().reset_index(drop=True))
    pred_2023.columns = ["Kota", "TPT Aktual", "Pred LR", "Pred RF", "Pred XGB"]
    st.dataframe(pred_2023.round(4), use_container_width=True)

    # ── Interpolasi Provinsi ───────────────────────────────────────────────
    st.markdown('<div class="section-title">[REVISI] Interpolasi TPT Provinsi Jawa Tengah (Berbobot Angkatan Kerja BPS)</div>', unsafe_allow_html=True)

    df_prov = res["df_prov"]
    if df_prov is None:
        st.warning("Upload file CSV Angkatan Kerja BPS (6 file, 2018–2023) di sidebar untuk mengaktifkan fitur ini.")
    else:
        st.dataframe(df_prov.round(4), use_container_width=True)

        # Error
        err_rows = []
        for col, label in [("TPT_Pred_LR_Provinsi","Regresi Linear"),
                           ("TPT_Pred_RF_Provinsi","Random Forest"),
                           ("TPT_Pred_XGB_Provinsi","XGBoost")]:
            mae_p  = np.mean(np.abs(df_prov["TPT_Aktual_Provinsi"] - df_prov[col]))
            mape_p = np.mean(np.abs((df_prov["TPT_Aktual_Provinsi"] - df_prov[col]) /
                                     df_prov["TPT_Aktual_Provinsi"])) * 100
            err_rows.append({"Model": label, "MAE Provinsi": round(mae_p,4), "MAPE Provinsi (%)": round(mape_p,2)})
        st.dataframe(pd.DataFrame(err_rows), use_container_width=True)

        # Grafik
        fig_prov, ax_prov = plt.subplots(figsize=(10, 5))
        ax_prov.plot(df_prov["Tahun"], df_prov["TPT_Aktual_Provinsi"],
                     marker="o", linewidth=2.5, label="Aktual", color="black")
        ax_prov.plot(df_prov["Tahun"], df_prov["TPT_Pred_LR_Provinsi"],
                     marker="s", linestyle="--", label="Regresi Linear", color="gray")
        ax_prov.plot(df_prov["Tahun"], df_prov["TPT_Pred_RF_Provinsi"],
                     marker="^", linestyle="--", label="Random Forest", color="steelblue")
        ax_prov.plot(df_prov["Tahun"], df_prov["TPT_Pred_XGB_Provinsi"],
                     marker="D", linestyle="--", label="XGBoost", color="darkorange")
        ax_prov.set_title("TPT Provinsi Jawa Tengah: Aktual vs Prediksi\n(Interpolasi Berbobot Angkatan Kerja BPS)",
                          fontsize=12, fontweight="bold")
        ax_prov.set_xlabel("Tahun"); ax_prov.set_ylabel("TPT (%)")
        ax_prov.legend(); ax_prov.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_prov); plt.close()

        # Download
        st.download_button("Unduh Tabel Interpolasi Provinsi (.csv)",
            df_prov.to_csv(index=False).encode("utf-8"),
            "tpt_provinsi_interpolasi.csv", "text/csv", use_container_width=True)
