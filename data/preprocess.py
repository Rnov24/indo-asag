"""
Preprocessing: Raw Excel → Clean Parquet
=========================================
Converts the 4 raw Excel files from Rahutomo & Ari Roshinta (2018)
into a single, clean parquet file ready for ASAG experiments.

Raw data structure:
    4 Excel files (Lifestyle, Olahraga, Politik, Teknologi)
    Each has sheets: Soal, No.1 to No.10, Nilai Rata
    - Soal:    question text + reference answer
    - No.X:    student answers + 3 manual scores + similarity metrics

Output: data/raw/dataset_master_autoscoring.parquet
    Clean columns: topic, question_id, question_text, reference_answer,
                   student_id, student_answer, score_m1, score_m2, score_m3,
                   score_manual_avg, score_manual_std

License: CC BY 4.0 — Rahutomo & Ari Roshinta (2018)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

RAW_DIR = Path(__file__).parent / "raw" / "Indonesian Query Answering Dataset for Online Essay Test System"
OUTPUT_PATH = Path(__file__).parent / "processed" / "dataset_master_autoscoring.parquet"

TOPICS = ["Lifestyle", "Olahraga", "Politik", "Teknologi"]

# Column mapping from raw Excel (unnamed) to clean names
# Raw sheet layout (No.X): 
#   Col 1: row number, Col 2: student_id, Col 3: answer
#   Col 4: Manual 1, Col 5: Manual 2, Col 6: Manual 3
#   Col 7-12: similarity metrics (Cos, Euc, Jac, Cos Stemm, Euc Stemm, Jacc Stemm)
#   Col 13: Rata Manual, Col 14+: Error columns


# ============================================================
# Step 1: Parse Questions & Reference Answers from "Soal" sheet
# ============================================================

def parse_questions(filepath: str) -> dict[int, dict]:
    """Extract question text and reference answer from the Soal sheet.
    
    Returns:
        Dict mapping question_id → {"question_text": str, "reference_answer": str}
    """
    df = pd.read_excel(filepath, sheet_name="Soal", header=None)
    
    questions = {}
    # Soal sheet layout: col 1 = No, col 2 = Soal, col 3 = Kunci Jawaban
    # First row (index 0) is header, data starts at index 1
    for _, row in df.iterrows():
        # Skip header and NaN rows
        q_num = row.iloc[1]
        if pd.isna(q_num) or not isinstance(q_num, (int, float)):
            try:
                q_num = int(q_num)
            except (ValueError, TypeError):
                continue
        
        q_num = int(q_num)
        questions[q_num] = {
            "question_text": str(row.iloc[2]).strip(),
            "reference_answer": str(row.iloc[3]).strip(),
        }
    
    return questions


# ============================================================
# Step 2: Parse Student Answers from "No.X" sheets
# ============================================================

def parse_answers(filepath: str, question_id: int) -> pd.DataFrame:
    """Extract student answers and manual scores from a single question sheet.
    
    Returns:
        DataFrame with columns: student_id, student_answer, score_m1, score_m2, score_m3
    """
    df = pd.read_excel(filepath, sheet_name=f"No.{question_id}", header=None)
    
    # Find the header row (contains "No" or "Siswa")
    header_row = None
    for i, row in df.iterrows():
        vals = [str(v).strip() for v in row.values if pd.notna(v)]
        if "Siswa" in vals or "No" in vals:
            header_row = i
            break
    
    if header_row is None:
        print(f"  WARNING: Could not find header in No.{question_id}")
        return pd.DataFrame()
    
    # Data starts after the header row
    data_rows = []
    for i in range(header_row + 1, len(df)):
        row = df.iloc[i]
        
        # Column indices (0-indexed):
        #   1: row number, 2: student_id, 3: answer
        #   4: Manual 1, 5: Manual 2, 6: Manual 3
        student_id = row.iloc[2]
        answer = row.iloc[3]
        m1 = row.iloc[4]
        m2 = row.iloc[5]
        m3 = row.iloc[6]
        rata = row.iloc[13] if len(row) > 13 else np.nan
        
        # Skip rows with missing essential data
        if pd.isna(answer) or pd.isna(student_id):
            continue
        if str(answer).strip() == "" or str(answer).strip().lower() == "nan":
            continue
        
        # Convert scores to float, handle missing
        def safe_float(x):
            try:
                return float(x)
            except (ValueError, TypeError):
                return np.nan
        
        data_rows.append({
            "student_id": str(student_id).strip(),
            "student_answer": str(answer).strip(),
            "score_m1": safe_float(m1),
            "score_m2": safe_float(m2),
            "score_m3": safe_float(m3),
            "score_excel_avg": safe_float(rata),
        })
    
    return pd.DataFrame(data_rows)


# ============================================================
# Step 3: Combine All Topics into Master DataFrame
# ============================================================

def build_master_dataset() -> pd.DataFrame:
    """Process all 4 topic Excel files into a single clean DataFrame."""
    
    all_rows = []
    stats = {"total_raw": 0, "dropped_empty": 0, "dropped_no_score": 0}
    
    for topic in TOPICS:
        filepath = RAW_DIR / f"Analisis_Essay_Grading_{topic}.xlsx"
        
        if not filepath.exists():
            print(f"WARNING: {filepath} not found, skipping.")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing: {topic}")
        print(f"{'='*50}")
        
        # Get questions for this topic
        questions = parse_questions(str(filepath))
        print(f"  Questions found: {len(questions)}")
        
        for q_id, q_info in sorted(questions.items()):
            answers_df = parse_answers(str(filepath), q_id)
            stats["total_raw"] += len(answers_df)
            
            if answers_df.empty:
                continue
            
            # Add question metadata
            answers_df["topic"] = topic
            answers_df["question_id"] = q_id
            answers_df["question_text"] = q_info["question_text"]
            answers_df["reference_answer"] = q_info["reference_answer"]
            
            all_rows.append(answers_df)
            print(f"  Q{q_id}: {len(answers_df)} answers")
    
    # Combine
    df = pd.concat(all_rows, ignore_index=True)
    
    # ============================================================
    # Step 4: Compute Average Score & Clean
    # ============================================================
    
    # Compute average across available raters
    score_cols = ["score_m1", "score_m2", "score_m3"]
    # Gunakan Rata Manual dari excel secara langsung, dengan mean rater sebagai cadangan
    df["score_manual_avg"] = df["score_excel_avg"].fillna(df[score_cols].mean(axis=1))
    df["score_manual_std"] = df[score_cols].std(axis=1)
    df["n_raters"] = df[score_cols].notna().sum(axis=1)
    
    # Drop rows with no valid scores
    no_score = df["score_manual_avg"].isna()
    stats["dropped_no_score"] = no_score.sum()
    df = df[~no_score]
    
    # Drop rows with empty answers (after stripping)
    empty_answer = df["student_answer"].str.strip() == ""
    stats["dropped_empty"] = empty_answer.sum()
    df = df[~empty_answer]
    
    # Reorder columns
    df = df[[
        "topic", "question_id", "question_text", "reference_answer",
        "student_id", "student_answer",
        "score_m1", "score_m2", "score_m3",
        "score_manual_avg", "score_manual_std", "n_raters",
    ]].reset_index(drop=True)
    
    return df, stats


# ============================================================
# Step 5: Validation & Summary
# ============================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """Print validation summary of the processed dataset."""
    
    print(f"\n{'='*60}")
    print("DATASET VALIDATION REPORT")
    print(f"{'='*60}")
    
    print(f"\n  Total rows:        {len(df)}")
    print(f"  Topics:            {df['topic'].nunique()} ({', '.join(df['topic'].unique())})")
    print(f"  Questions:         {df['question_id'].nunique()}")
    print(f"  Unique students:   {df['student_id'].nunique()}")
    
    print(f"\n  Score statistics:")
    print(f"    Mean:   {df['score_manual_avg'].mean():.2f}")
    print(f"    Std:    {df['score_manual_avg'].std():.2f}")
    print(f"    Min:    {df['score_manual_avg'].min():.2f}")
    print(f"    Max:    {df['score_manual_avg'].max():.2f}")
    print(f"    Median: {df['score_manual_avg'].median():.2f}")
    
    print(f"\n  Rater coverage:")
    print(f"    3 raters: {(df['n_raters'] == 3).sum()} rows ({(df['n_raters'] == 3).mean():.1%})")
    print(f"    2 raters: {(df['n_raters'] == 2).sum()} rows ({(df['n_raters'] == 2).mean():.1%})")
    print(f"    1 rater:  {(df['n_raters'] == 1).sum()} rows ({(df['n_raters'] == 1).mean():.1%})")
    
    print(f"\n  Per-topic breakdown:")
    for topic in df["topic"].unique():
        sub = df[df["topic"] == topic]
        print(f"    {topic:12s}: {len(sub):4d} rows, "
              f"avg score = {sub['score_manual_avg'].mean():.1f}")
    
    print(f"\n  Answer length (words):")
    wc = df["student_answer"].str.split().str.len()
    print(f"    Mean: {wc.mean():.1f}, Median: {wc.median():.0f}, "
          f"Min: {wc.min()}, Max: {wc.max()}")
    
    # Data quality checks
    print(f"\n  Quality checks:")
    print(f"    Empty answers: {(df['student_answer'].str.strip() == '').sum()}")
    print(f"    NaN scores:    {df['score_manual_avg'].isna().sum()}")
    print(f"    Score < 0:     {(df['score_manual_avg'] < 0).sum()}")
    print(f"    Score > 100:   {(df['score_manual_avg'] > 100).sum()}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Indo-ASAG Data Preprocessing")
    print("=" * 60)
    print(f"Raw data: {RAW_DIR}")
    print(f"Output:   {OUTPUT_PATH}")
    
    df, stats = build_master_dataset()
    
    print(f"\n  Processing stats:")
    print(f"    Raw rows:        {stats['total_raw']}")
    print(f"    Dropped (empty): {stats['dropped_empty']}")
    print(f"    Dropped (score): {stats['dropped_no_score']}")
    print(f"    Final rows:      {len(df)}")
    
    validate_dataset(df)
    
    # Save
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n  [OK] Saved to: {OUTPUT_PATH}")
    print(f"  File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
