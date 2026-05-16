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
# # Eksperimen 6: Late Fusion (Regresi Ridge) Sebagai Solusi Utama
#
# **GEMASTIK KTI 2026** | Tim Peneliti
#
# Sebagai alternatif dari penggabungan representasi vektor awal, metodologi ini hanya menggabungkan prediksi akhir (skor) keluaran dari model IndoBERT dan model fitur Sastrawi. Penggabungan tersebut difasilitasi menggunakan algoritma Regresi Ridge untuk menyeimbangkan serta mencari bobot paling optimal secara analitis.

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
from indo_asag.evaluation import run_multi_seed
from indo_asag.utils import set_seed, load_config

config = load_config(os.path.join(REPO_ROOT, "configs", "base.yaml"))
SEEDS = config["seeds"]
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "prelim")
PREDS_DIR = os.path.join(RESULTS_DIR, "predictions")
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")

# %% [markdown]
# ## 1. Pemuatan Dataset dan Komponen Prasyarat
#
# Eksperimen ini mewajibkan Ekperimen 3 (IndoBERT) dan Eksperimen 4 (Handcrafted) untuk dijalankan terlebih dahulu, karena membutuhkan prediksi OOF dari kedua model.

# %%
DATA_PATH = os.path.join(REPO_ROOT, config["data"]["path"])
df = load_dataset(DATA_PATH)

bert_oof_preds = {}
hc_oof_preds = {}

for s in SEEDS:
    bert_oof_preds[s] = np.load(os.path.join(PREDS_DIR, f"exp3_indobert_oof_seed{s}.npy"))
    hc_oof_preds[s] = np.load(os.path.join(PREDS_DIR, f"exp4_hc_oof_seed{s}.npy"))

# %% [markdown]
# ## 2. Eksekusi Eksperimen 6

# %%
from indo_asag.models import LateFusion
from sklearn.linear_model import Ridge

print("\n" + "=" * 60)
print("EXP 6: Late Fusion (Regresi Ridge)")
print("=" * 60)

exp6_oof_preds = {s: np.zeros(len(df)) for s in SEEDS}

def exp6_predict(train_df, val_df, fold, seed=42):
    X_tr = np.column_stack([bert_oof_preds[seed][train_df.index], hc_oof_preds[seed][train_df.index]])
    X_va = np.column_stack([bert_oof_preds[seed][val_df.index], hc_oof_preds[seed][val_df.index]])

    ridge = Ridge(alpha=config["model"]["ridge"]["alpha"])
    ridge.fit(X_tr, train_df["score_norm"].values)
    
    joblib.dump(ridge, os.path.join(CHECKPOINT_DIR, f"ridge_latefusion_seed{seed}_fold{fold}.pkl"))
    
    preds = ridge.predict(X_va)
    exp6_oof_preds[seed][val_df.index] = preds

    if fold == 0 and seed == SEEDS[0]:
        w = ridge.coef_
        total = abs(w[0]) + abs(w[1])
        print(f"  Bobot otomatis: IndoBERT={w[0]/total:.1%}, Sastrawi(HC)={w[1]/total:.1%}")

    return preds

exp6_summary = run_multi_seed(df, exp6_predict, seeds=SEEDS)
print(f"  QWK: {exp6_summary['QWK']}, Pearson: {exp6_summary['Pearson']}")

# %% [markdown]
# ## 3. Penyimpanan Prediksi (Out-of-Fold)

# %%
print("\nMenyimpan array prediksi OOF ke disk...")
for s in SEEDS:
    np.save(os.path.join(PREDS_DIR, f"exp6_late_fusion_oof_seed{s}.npy"), exp6_oof_preds[s])
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
            os.system(f'cd {REPO_ROOT} && git commit -m "chore(auto): menyimpan prediksi Eksperimen 6 Late Fusion"')
            
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

