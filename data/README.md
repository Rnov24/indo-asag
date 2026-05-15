# Data Directory

## Dataset: Indonesian Short Answer Grading

| Property | Value |
|---|---|
| Source | Rahutomo & Ari Roshinta (2018) |
| License | **CC BY 4.0** (included in repo) |
| Size | 2,162 short answer pairs |
| Topics | 4 (Lifestyle, Olahraga, Politik, Teknologi) |
| Questions | 10 |
| Score Range | 0–100 (3 human raters, averaged) |
| Published | Mendeley Data |

## Structure

```
data/
├── raw/                               # Original Excel files (untouched)
│   └── Indonesian Query Answering.../
│       ├── Analisis_Essay_Grading_Lifestyle.xlsx
│       ├── Analisis_Essay_Grading_Olahraga.xlsx
│       ├── Analisis_Essay_Grading_Politik.xlsx
│       └── Analisis_Essay_Grading_Teknologi.xlsx
├── processed/                         # Clean output
│   └── dataset_master_autoscoring.parquet
└── preprocess.py                      # raw → processed
```

## Regenerate

```bash
python data/preprocess.py
```

## Columns (processed parquet)

| Column | Description |
|---|---|
| `topic` | Topic category |
| `question_id` | Question identifier (1–10) |
| `question_text` | The question text |
| `reference_answer` | Reference/key answer |
| `student_id` | Student identifier |
| `student_answer` | Student's answer |
| `score_m1` | Score from rater 1 (0–100) |
| `score_m2` | Score from rater 2 (0–100) |
| `score_m3` | Score from rater 3 (0–100) |
| `score_manual_avg` | Average score from 3 raters |
| `score_manual_std` | Std dev across 3 raters |
| `n_raters` | Number of valid rater scores |
