"""GitHub auto-push utility for Google Colab environments."""

import os

def auto_push_to_github(commit_msg: str, in_colab: bool, repo_root: str):
    """Automatically pushes changes to GitHub if running in Google Colab.
    
    Args:
        commit_msg: The commit message to use.
        in_colab: Boolean indicating if currently running in Colab.
        repo_root: Absolute path to the repository root.
    """
    if not in_colab:
        print("[INFO] Tidak berjalan di Google Colab. Melewati proses auto-push ke GitHub.")
        return

    from google.colab import userdata
    try:
        GH_TOKEN = userdata.get('GITHUB_TOKEN')
    except userdata.SecretNotFoundError:
        print("Peringatan: Kunci rahasia 'GITHUB_TOKEN' tidak ditemukan di Google Colab.")
        return
    except Exception as e:
        print(f"Peringatan: Autentikasi rahasia tertunda/terhenti ({type(e).__name__}). Melanjutkan eksekusi tanpa auto-push GitHub.")
        return

    if not GH_TOKEN:
        return

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
        os.system(f'cd {repo_root} && git add notebooks/*.ipynb results/prelim/metrics/*.csv results/prelim/predictions/*.npy checkpoints/*')
        os.system(f'cd {repo_root} && git commit -m "{commit_msg}"')
        
        # Pengiriman pembaruan (Push)
        repo_url = f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{GH_REPO}.git"
        
        # Menarik perubahan terbaru (Pull) dengan rebase untuk mencegah penolakan (non-fast-forward)
        os.system(f'cd {repo_root} && git pull {repo_url} main --rebase > /dev/null 2>&1')
        
        push_cmd = f'cd {repo_root} && git push {repo_url} main'
        
        # Eksekusi (menyembunyikan output tautan agar token tidak terekspos di log Colab)
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
