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
# # Laporan Akhir dan Uji Signifikansi Statistik
#
# **GEMASTIK KTI 2026** | Tim Peneliti
#
# *Notebook* ini difungsikan secara murni untuk evaluasi dan pelaporan. Tidak ada proses pelatihan model (training) yang dilakukan di sini. Seluruh metrik evaluasi dihitung secara langsung dari hasil prediksi (*Out-of-Fold* arrays) yang dimuat dari *repository* lokal.
#
# Untuk menjamin bahwa peningkatan performa antar model bukan disebabkan oleh variansi acak, pengujian signifikansi statistik diimplementasikan menggunakan Paired T-Test (N=2162).

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
from scipy import stats

from indo_asag.data import load_dataset
from indo_asag.evaluation.metrics import evaluate
from indo_asag.utils import load_config

config = load_config(os.path.join(REPO_ROOT, "configs", "base.yaml"))
SEEDS = config["seeds"]
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "prelim")
PREDS_DIR = os.path.join(RESULTS_DIR, "predictions")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# %% [markdown]
# ## 1. Pemuatan Data dan Prediksi

# %%
DATA_PATH = os.path.join(REPO_ROOT, config["data"]["path"])
df = load_dataset(DATA_PATH)
y_true = df["score_norm"].values

# Struktur data untuk menyimpan prediksi semua seed per eksperimen
exp_preds = {
    "Exp 1: SVR (Cos, Euc, Jac)": [],
    "Exp 2: Frozen SBERT": [],
    "Exp 3: FT IndoBERT (Ref+A)": [],
    "Exp 4: Sastrawi HC + SVR": [],
    "Exp 5: Naive Concat (GAGAL)": [],
    "Exp 6: Late Fusion (SUKSES)": []
}

file_mapping = {
    "Exp 1: SVR (Cos, Euc, Jac)": "exp1_lexical_oof_seed{s}.npy",
    "Exp 2: Frozen SBERT": "exp2_sbert_oof_seed{s}.npy",
    "Exp 3: FT IndoBERT (Ref+A)": "exp3_indobert_oof_seed{s}.npy",
    "Exp 4: Sastrawi HC + SVR": "exp4_hc_oof_seed{s}.npy",
    "Exp 5: Naive Concat (GAGAL)": "exp5_concat_oof_seed{s}.npy",
    "Exp 6: Late Fusion (SUKSES)": "exp6_late_fusion_oof_seed{s}.npy"
}

# Memuat semua array prediksi dari disk
for exp_name, file_pattern in file_mapping.items():
    try:
        for s in SEEDS:
            filepath = os.path.join(PREDS_DIR, file_pattern.format(s=s))
            exp_preds[exp_name].append(np.load(filepath))
        print(f"[OK] Memuat prediksi {exp_name}")
    except FileNotFoundError:
        print(f"[WARNING] Prediksi untuk {exp_name} tidak ditemukan. Harap jalankan script eksperimennya terlebih dahulu.")
        del exp_preds[exp_name]

# %% [markdown]
# ## 2. Laporan Ringkasan Kinerja Keseluruhan

# %%
print("\n" + "=" * 60)
print("LAPORAN RINGKASAN KINERJA (SUMMARY REPORT)")
print("=" * 60)

rows = []

for exp_name, preds_list in exp_preds.items():
    metrics_list = []
    for preds in preds_list:
        metrics_list.append(evaluate(y_true, preds))
    
    # Hitung rata-rata dan standar deviasi lintas seed
    mean_metrics = pd.DataFrame(metrics_list).mean().to_dict()
    std_metrics = pd.DataFrame(metrics_list).std().to_dict()
    
    rows.append({
        "Eksperimen": exp_name,
        "QWK": f"{mean_metrics['QWK']:.4f} +/- {std_metrics['QWK']:.4f}",
        "Pearson": f"{mean_metrics['Pearson']:.4f} +/- {std_metrics['Pearson']:.4f}",
        "RMSE": f"{mean_metrics['RMSE']:.4f} +/- {std_metrics['RMSE']:.4f}",
        "MAE": f"{mean_metrics['MAE']:.4f} +/- {std_metrics['MAE']:.4f}",
        "Seeds": str(len(SEEDS)),
    })

summary_df = pd.DataFrame(rows)
print("\n" + summary_df.to_string(index=False))

# Penyimpanan hasil evaluasi
summary_df.to_csv(os.path.join(METRICS_DIR, "prelim_results.csv"), index=False)

# %% [markdown]
# ## 3. Eksperimen 8: Uji Signifikansi Statistik (Paired T-Test)
#
# Pengujian signifikansi mengevaluasi Galat Absolut (Absolute Error) pada tingkat instans (N=2162) yang diturunkan dari agregat prediksi lintas lima iterasi acak.

# %%
print("\n" + "=" * 60)
print("EXP 8: Uji Signifikansi Statistik (Paired T-Test N=2162)")
print("=" * 60)

def get_instance_abs_error(exp_name):
    if exp_name not in exp_preds:
        return None
    # Agregasi rata-rata prediksi lintas 5 seed
    avg_preds = np.mean(exp_preds[exp_name], axis=0)
    return np.abs(avg_preds - y_true)

def perform_ttest(name_a, name_b):
    err_a = get_instance_abs_error(name_a)
    err_b = get_instance_abs_error(name_b)
    
    if err_a is None or err_b is None:
        print(f"\n[Dilewati] Data tidak lengkap untuk komparasi: {name_a} vs {name_b}")
        return
        
    stat, p = stats.ttest_rel(err_a, err_b)
    sig = "SIGNIFIKAN (Tolak H0)" if p < 0.05 else "TIDAK SIGNIFIKAN (Terima H0)"
    print(f"\nKomparasi: {name_a} vs {name_b}")
    print(f"  T-Statistic : {stat:.4f}")
    print(f"  P-Value     : {p:.4e}")
    print(f"  Kesimpulan  : {sig}")

perform_ttest("Exp 6: Late Fusion (SUKSES)", "Exp 4: Sastrawi HC + SVR")
perform_ttest("Exp 4: Sastrawi HC + SVR", "Exp 3: FT IndoBERT (Ref+A)")
perform_ttest("Exp 3: FT IndoBERT (Ref+A)", "Exp 2: Frozen SBERT")

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
            os.system(f'cd {REPO_ROOT} && git commit -m "chore(auto): update laporan metrik dan uji statistik akhir"')
            
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

