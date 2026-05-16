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
# # Eksperimen 4: Model Leksikal (Fitur Morfologis Sastrawi dan SVR)
#
# **GEMASTIK KTI 2026** | Tim Peneliti
#
# Eksperimen ini mengevaluasi 11 fitur linguistik sederhana yang digabungkan dengan algoritma SVR. Jika model regresi konvensional ini mampu mengimbangi arsitektur IndoBERT yang memiliki 110 juta parameter, hal tersebut akan menjadi bukti empiris yang valid bahwa morfologi Bahasa Indonesia memiliki sinyal penilaian otomatis yang sangat kuat.

# %% [markdown]
# ## 0. Persiapan Lingkungan dan Konfigurasi

# %%
import sys
import os

try:
    import google.colab
    IN_COLAB = True
    print("Lingkungan Eksekusi: Google Colab")
    if not os.path.exists("/content/indo-asag"):
        os.system("git clone https://github.com/Rnov24/indo-asag.git /content/indo-asag")
    os.system("pip install -q -e /content/indo-asag[all]")
    REPO_ROOT = "/content/indo-asag"
except ImportError:
    IN_COLAB = False
    try:
        REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except NameError:
        REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
    print(f"Lingkungan Eksekusi: Lokal ({REPO_ROOT})")

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from indo_asag.data import load_dataset
from indo_asag.features import extract_features_df, get_feature_names
from indo_asag.evaluation import run_multi_seed
from indo_asag.utils import set_seed, load_config

config = load_config(os.path.join(REPO_ROOT, "configs", "base.yaml"))
SEEDS = config["seeds"]
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "prelim")
PREDS_DIR = os.path.join(RESULTS_DIR, "predictions")

# %% [markdown]
# ## 1. Pemuatan Dataset dan Ekstraksi Fitur Sastrawi

# %%
DATA_PATH = os.path.join(REPO_ROOT, config["data"]["path"])
df = load_dataset(DATA_PATH)

print("\nMengekstrak 11 fitur Sastrawi (Handcrafted)...")
feat_df = extract_features_df(df)
hc_cols = get_feature_names()

for col in hc_cols:
    df[col] = feat_df[col]

# Save the extracted features so we don't have to extract them again in Exp 5
df[hc_cols].to_csv(os.path.join(PREDS_DIR, "features_hc.csv"), index=False)
print("[OK] Fitur berhasil diekstrak dan disimpan.")

# %% [markdown]
# ## 2. Eksekusi Eksperimen 4

# %%
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

print("\n" + "=" * 60)
print("EXP 4: Model Leksikal (Fitur Morfologis Sastrawi dan SVR)")
print("=" * 60)

hc_oof_preds = {s: np.zeros(len(df)) for s in SEEDS}

def exp4_predict(train_df, val_df, fold, seed=42):
    X_tr = train_df[hc_cols].values
    X_va = val_df[hc_cols].values
    
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
    
    preds = svr.predict(X_va_s)
    hc_oof_preds[seed][val_df.index] = preds
    return preds

exp4_summary = run_multi_seed(df, exp4_predict, seeds=SEEDS)
print(f"  QWK: {exp4_summary['QWK']}")

# %% [markdown]
# ## 3. Penyimpanan Prediksi (Out-of-Fold)

# %%
print("\nMenyimpan array prediksi OOF ke disk...")
for s in SEEDS:
    np.save(os.path.join(PREDS_DIR, f"exp4_hc_oof_seed{s}.npy"), hc_oof_preds[s])
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
            os.system(f'cd {REPO_ROOT} && git commit -m "chore(auto): menyimpan prediksi Eksperimen 4 Handcrafted SVR"')
            
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

