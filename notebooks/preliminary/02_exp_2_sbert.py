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
# # Eksperimen 2: Baseline Embedding (Frozen SBERT)
#
# **GEMASTIK KTI 2026** | Tim Peneliti
#
# Eksperimen ini menguji apakah model bahasa pralatih (pretrained) yang menangkap representasi semantik dasar sudah cukup memadai untuk melakukan penilaian, tanpa perlu penyesuaian khusus (fine-tuning). Model SBERT multibahasa digunakan untuk mengekstrak vektor semantik berdimensi 384 dari jawaban siswa dan kunci referensi. Serupa dengan Eksperimen 1, tiga metrik jarak (Cosine, Euclidean, dan Continuous Jaccard) dikalkulasi dari vektor embedding tersebut, kemudian diintegrasikan ke dalam algoritma SVR.

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

from indo_asag.data import load_dataset
from indo_asag.evaluation import run_multi_seed
from indo_asag.utils import set_seed, load_config

config = load_config(os.path.join(REPO_ROOT, "configs", "base.yaml"))
SEEDS = config["seeds"]
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "prelim")
PREDS_DIR = os.path.join(RESULTS_DIR, "predictions")

# %% [markdown]
# ## 1. Pemuatan Dataset

# %%
DATA_PATH = os.path.join(REPO_ROOT, config["data"]["path"])
df = load_dataset(DATA_PATH)

# %% [markdown]
# ## 2. Eksekusi Eksperimen 2

# %%
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

print("\n" + "=" * 60)
print("EXP 2: Frozen SBERT (Cosine, Euclidean, Jaccard + SVR)")
print("=" * 60)

sbert_model = SentenceTransformer(config["features"]["sbert_model"])

print("Mengubah jawaban menjadi embedding (encoding)...")
ans_emb = sbert_model.encode(df["student_answer"].tolist(), batch_size=64,
                              normalize_embeddings=True, show_progress_bar=True)
ref_emb = sbert_model.encode(df["reference_answer"].tolist(), batch_size=64,
                              normalize_embeddings=True, show_progress_bar=True)

# Compute distances
cos_sim = np.sum(ans_emb * ref_emb, axis=1)
euc_dist = np.linalg.norm(ans_emb - ref_emb, axis=1)
tanimoto_jac = cos_sim / (2.0 - cos_sim)

df["sbert_cos"] = cos_sim
df["sbert_euc"] = euc_dist
df["sbert_jac"] = tanimoto_jac

def exp2_predict(train_df, val_df, fold, seed=42):
    features = ["sbert_cos", "sbert_euc", "sbert_jac"]
    X_tr = train_df[features].values
    X_va = val_df[features].values
    
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

exp2_summary = run_multi_seed(df, exp2_predict, seeds=SEEDS)
print(f"  QWK: {exp2_summary['QWK']}")

# %% [markdown]
# ## 3. Penyimpanan Prediksi (Out-of-Fold)

# %%
print("\nMenyimpan array prediksi OOF ke disk...")
for s, preds in exp2_summary["_preds"].items():
    np.save(os.path.join(PREDS_DIR, f"exp2_sbert_oof_seed{s}.npy"), preds)
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
            os.system(f'cd {REPO_ROOT} && git commit -m "chore(auto): menyimpan prediksi Eksperimen 2 SBERT"')
            
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

