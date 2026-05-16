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
# # Eksperimen 3: IndoBERT Fine-Tuned (Referensi dan Jawaban)
#
# **GEMASTIK KTI 2026** | Tim Peneliti
#
# Format masukan model: `[CLS] kunci_jawaban [SEP] jawaban_siswa [SEP]`
#
# Pendekatan ini memberikan model bahasa akses penuh ke kunci jawaban secara bersamaan. Hal ini memastikan model dievaluasi pada kondisi yang setara dengan model fitur leksikal, di mana keduanya bertugas mengomparasikan kedua teks secara langsung.

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
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")

# %% [markdown]
# ## 1. Pemuatan Dataset

# %%
DATA_PATH = os.path.join(REPO_ROOT, config["data"]["path"])
df = load_dataset(DATA_PATH)

# %% [markdown]
# ## 2. Eksekusi Eksperimen 3

# %%
from indo_asag.models.bert_regressor import BertRegressor

print("\n" + "=" * 60)
print("EXP 3: IndoBERT Fine-Tuned (Referensi dan Jawaban)")
print("=" * 60)

bert = BertRegressor(
    model_name=config["model"]["bert"]["name"],
    dropout=config["model"]["bert"]["dropout"],
)

bert_oof_preds = {s: np.zeros(len(df)) for s in SEEDS}
bert_oof_embs = {s: np.zeros((len(df), 768)) for s in SEEDS}

def exp3_predict(train_df, val_df, fold, seed=42):
    preds, embs = bert.train_fold(
        train_df, val_df, fold,
        text_a_col="reference_answer",
        text_b_col="student_answer",
        epochs=config["model"]["bert"]["epochs"],
        batch_size=config["model"]["bert"]["batch_size"],
        lr=config["model"]["bert"]["learning_rate"],
        save_path=os.path.join(CHECKPOINT_DIR, f"indobert_seed{seed}_fold{fold}.pt")
    )
    bert_oof_preds[seed][val_df.index] = preds
    bert_oof_embs[seed][val_df.index] = embs
    return preds

exp3_summary = run_multi_seed(df, exp3_predict, seeds=SEEDS)
print(f"  QWK: {exp3_summary['QWK']}, Pearson: {exp3_summary['Pearson']}")

# %% [markdown]
# ## 3. Penyimpanan Prediksi dan Embeddings (Out-of-Fold)

# %%
print("\nMenyimpan array prediksi dan embeddings OOF ke disk...")
for s in SEEDS:
    np.save(os.path.join(PREDS_DIR, f"exp3_indobert_oof_seed{s}.npy"), bert_oof_preds[s])
    np.save(os.path.join(PREDS_DIR, f"exp3_indobert_emb_seed{s}.npy"), bert_oof_embs[s])
print("[OK] Prediksi dan embeddings berhasil disimpan.")

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
            os.system(f'cd {REPO_ROOT} && git commit -m "chore(auto): menyimpan prediksi Eksperimen 3 IndoBERT"')
            
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

