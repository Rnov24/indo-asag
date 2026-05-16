# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Eksperimen 1: Baseline Leksikal (TF-IDF dan Metrik Jarak Klasik)
#
# **GEMASTIK KTI 2026** | Tim Peneliti
#
# Eksperimen pertama ini menerapkan pendekatan leksikal dasar. Representasi teks dibangun menggunakan pembobotan TF-IDF untuk menangkap kepentingan kata (bag-of-words). Tiga metrik jarak (Cosine Similarity, Euclidean Distance, dan Jaccard Similarity) diekstraksi dari vektor teks tersebut, kemudian digabungkan ke dalam algoritma Support Vector Regression (SVR) untuk memprediksi nilai akhir. Eksperimen ini berfungsi sebagai standar acuan paling mendasar.

# %% [markdown]
# ## 0. Persiapan Lingkungan dan Konfigurasi
#
# Bagian ini menginisialisasi lingkungan eksekusi, memuat pustaka yang dibutuhkan, serta menyiapkan konfigurasi dasar.

# %%
import sys
import os

# Deteksi lingkungan eksekusi
try:
    import google.colab
    IN_COLAB = True
    print("Lingkungan Eksekusi: Google Colab")

    # Kloning repositori jika belum tersedia
    if not os.path.exists("/content/indo-asag"):
        os.system("git clone https://github.com/Rnov24/indo-asag.git /content/indo-asag")

    # Instalasi paket utama
    os.system("pip install -q -e /content/indo-asag[all]")
    REPO_ROOT = "/content/indo-asag"
except ImportError:
    IN_COLAB = False
    try:
        REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except NameError:
        REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
    print(f"Lingkungan Eksekusi: Lokal ({REPO_ROOT})")

# Penambahan direktori sumber ke sistem path
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from indo_asag.data import load_dataset
from indo_asag.evaluation import run_multi_seed
from indo_asag.utils import set_seed, load_config

# Konfigurasi
config = load_config(os.path.join(REPO_ROOT, "configs", "base.yaml"))
SEEDS = config["seeds"]
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "prelim")
PREDS_DIR = os.path.join(RESULTS_DIR, "predictions")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")

for d in [PREDS_DIR, METRICS_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# %% [markdown]
# ## 1. Pemuatan Dataset

# %%
DATA_PATH = os.path.join(REPO_ROOT, config["data"]["path"])
df = load_dataset(DATA_PATH)
print(f"Total data: {len(df)}")

# %% [markdown]
# ## 2. Eksekusi Eksperimen 1

# %%
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

print("\n" + "=" * 60)
print("EXP 1: Baseline Leksikal (Cosine, Euclidean, Jaccard + SVR)")
print("=" * 60)

def exp1_predict(train_df, val_df, fold, seed=42):
    all_texts = pd.concat([
        train_df["student_answer"], val_df["student_answer"],
        train_df["reference_answer"], val_df["reference_answer"]
    ])
    tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True)
    tfidf.fit(all_texts)
    
    cv = CountVectorizer(max_features=5000, binary=True)
    cv.fit(all_texts)
    
    def extract_sims(df_subset):
        va_t = tfidf.transform(df_subset["student_answer"])
        vr_t = tfidf.transform(df_subset["reference_answer"])
        va_c = cv.transform(df_subset["student_answer"])
        vr_c = cv.transform(df_subset["reference_answer"])
        
        cos = np.array([cosine_similarity(va_t[i], vr_t[i])[0, 0] for i in range(va_t.shape[0])])
        euc = np.array([euclidean_distances(va_t[i], vr_t[i])[0, 0] for i in range(va_t.shape[0])])
        jac = np.array([1 - pairwise_distances(va_c[i].toarray(), vr_c[i].toarray(), metric="jaccard")[0, 0] if va_c[i].nnz > 0 or vr_c[i].nnz > 0 else 1.0 for i in range(va_c.shape[0])])
        return np.column_stack([cos, euc, jac])

    X_tr = extract_sims(train_df)
    X_va = extract_sims(val_df)
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    
    svr = SVR(
        kernel=config["model"]["svr"]["kernel"],
        C=config["model"]["svr"]["C"],
        gamma=config["model"]["svr"]["gamma"],
        epsilon=config["model"]["svr"]["epsilon"]
    )
    svr.fit(X_tr_s, train_df["score_norm"].values)
    return svr.predict(X_va_s)

exp1_summary = run_multi_seed(df, exp1_predict, seeds=SEEDS)
print(f"  QWK: {exp1_summary['QWK']}")

# %% [markdown]
# ## 3. Penyimpanan Prediksi (Out-of-Fold)

# %%
print("\nMenyimpan array prediksi OOF ke disk...")
for s, preds in exp1_summary["_preds"].items():
    np.save(os.path.join(PREDS_DIR, f"exp1_lexical_oof_seed{s}.npy"), preds)
print("[OK] Prediksi berhasil disimpan.")

# %% [markdown]
# ## 4. Publikasi Otomatis ke GitHub

# %%
if IN_COLAB:
    from google.colab import userdata
    try:
        GH_TOKEN = userdata.get('GITHUB_TOKEN')
    except userdata.SecretNotFoundError:
        print("Peringatan: Kunci rahasia 'GITHUB_TOKEN' tidak ditemukan di Google Colab.")
        GH_TOKEN = None
    except Exception as e:
        print(f"Peringatan: Autentikasi rahasia tertunda/terhenti ({type(e).__name__}). Melanjutkan eksekusi tanpa auto-push GitHub.")
        GH_TOKEN = None

    if GH_TOKEN:
        print("\n" + "=" * 60)
        print("MENGIRIMKAN PEMBARUAN KE GITHUB (PUSH)")
        print("=" * 60)
        
        try:
            GH_USER = "Rnov24"
            GH_REPO = "indo-asag"
            GH_EMAIL = "rrrijal24@gmail.com"
            
            # Konfigurasi repositori lokal
            os.system(f'git config --global user.email "{GH_EMAIL}"')
            os.system(f'git config --global user.name "{GH_USER}"')
            
            # Staging dan Commit
            os.system(f'cd {REPO_ROOT} && git add notebooks/preliminary/*.ipynb results/prelim/metrics/*.csv results/prelim/predictions/*.npy checkpoints/*')
            os.system(f'cd {REPO_ROOT} && git commit -m "chore(auto): menyimpan prediksi Eksperimen 1 Leksikal"')
            
            # Pengiriman pembaruan (Push)
            repo_url = f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{GH_REPO}.git"
            
            # Menarik perubahan terbaru (Pull) dengan rebase
            os.system(f'cd {REPO_ROOT} && git pull {repo_url} main --rebase > /dev/null 2>&1')
            
            push_cmd = f'cd {REPO_ROOT} && git push {repo_url} main'
            
            # Eksekusi (menyembunyikan output)
            result = os.system(push_cmd + " > /dev/null 2>&1")
            
            if result == 0:
                print("[OK] Berhasil menyimpan dan mengirimkan hasil eksperimen ke repositori GitHub.")
                print("[INFO] Mengeksekusi penghentian otomatis (shutdown) runtime Colab untuk mengoptimalkan penggunaan sumber daya.")
                from google.colab import runtime
                runtime.unassign()
            else:
                print("[GAGAL] Proses pengiriman ke GitHub tidak berhasil. Harap periksa kembali token dan izin akses repositori.")
                
        except Exception as e:
            print(f"[KESALAHAN] Terjadi kendala saat berinteraksi dengan GitHub: {e}")

