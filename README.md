# Indo-ASAG: Indonesian Automated Short Answer Grading

Sistem penilaian jawaban singkat otomatis Bahasa Indonesia berbasis **Late Fusion** yang mengintegrasikan *fine-tuned* IndoBERT dengan fitur leksikal (Sastrawi stemmer).

## Quick Start

```bash
# Install (editable mode)
pip install -e ".[all]"

# Atau di Google Colab
!pip install -e "/content/indo-asag[all]"
```

## Struktur Repositori

```
├── notebooks/          # Interface utama (Colab-ready)
├── src/indo_asag/      # Python package modular
├── configs/            # YAML configs
├── data/               # Dataset (gitignored)
├── results/            # Output eksperimen
└── paper/              # LaTeX documents
```

## Menjalankan Eksperimen

Buka notebook di `notebooks/` secara berurutan:

| Notebook | Eksperimen |
|---|---|
| `01_prelim_baseline.ipynb` | TF-IDF + SBERT baselines |
| `02_prelim_deep.ipynb` | IndoBERT fine-tuning |
| `03_prelim_shallow.ipynb` | HC Sastrawi + SVR |
| `04_prelim_fusion.ipynb` | Naive Concat vs Late Fusion |
| `05_prelim_loqo.ipynb` | Cross-prompt evaluation |

## Sitasi

```bibtex
@misc{rijal2026indoasag,
  title={Indonesian ASAG via Late Fusion of Fine-Tuned IndoBERT and Morphological Features},
  author={Rijal},
  year={2026}
}
```
