
from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

# 1. Load dataset
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "resume_data.csv"

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# Preserve original dataset row identity BEFORE filtering
df["source_row_id"] = np.arange(len(df))

# Normalize column names
df.columns = (
    df.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.strip('"')
    .str.strip("'")
)

# 2. Select resume and job columns
resume_columns = [
    "career_objective",
    "skills",
    "degree_names",
    "major_field_of_studies",
    "positions",
    "responsibilities",
]

job_columns = [
    "role_positions",
    "educationaL_requirements",
    "experiencere_requirement",
    "skills_required",
    "responsibilities.1",
]

required = resume_columns + job_columns + ["matched_score"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        f"Available columns: {df.columns.tolist()}"
    )

# 3. Combine resume and job text
def combine_text(frame, columns):
    return (
        frame[columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )


df["resume_text"] = combine_text(df, resume_columns)
df["job_text"] = combine_text(df, job_columns)

df["matched_score"] = pd.to_numeric(
    df["matched_score"], errors="coerce"
)

# Filter invalid records while retaining source_row_id
df = df.dropna(subset=["matched_score"]).copy()

df = df[
    df["resume_text"].ne("") &
    df["job_text"].ne("")
].copy()

if len(df) < 3:
    raise ValueError("Not enough valid rows for evaluation.")

# 4. Use the same split settings as the TF-IDF baseline
_, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
)

test_df = test_df.copy()

print("MiniLM test samples:", len(test_df))

# 5. Load MiniLM
print("\nLoading MiniLM model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# 6. Generate resume embeddings
print("\nEncoding resumes...")

resume_embeddings = model.encode(
    test_df["resume_text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
)

# 7. Generate job-description embeddings
print("\nEncoding job descriptions...")

job_embeddings = model.encode(
    test_df["job_text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
)

# 8. Calculate cosine similarity
similarities = np.sum(
    resume_embeddings * job_embeddings,
    axis=1,
)

# Map cosine similarity from [-1, 1] to [0, 1]
predictions = np.clip(
    (similarities + 1) / 2,
    0,
    1,
)

# 9. Evaluate predictions
actual = test_df["matched_score"].to_numpy()

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

pearson = (
    pearsonr(actual, predictions).statistic
    if len(actual) >= 2
    and np.std(actual) > 0
    and np.std(predictions) > 0
    else np.nan
)

spearman = (
    spearmanr(actual, predictions).statistic
    if len(actual) >= 2
    else np.nan
)

print("\nMINILM RESULTS")
print("--------------------")
print("Test samples:", len(test_df))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("Pearson:", round(pearson, 4))
print("Spearman:", round(spearman, 4))

# 10. Save results, including original row IDs
results_dir = ROOT / "experiments" / "results"
results_dir.mkdir(parents=True, exist_ok=True)

output = test_df[
    ["source_row_id", "matched_score"]
].copy()

output["minilm_similarity"] = predictions

output.to_csv(
    results_dir / "minilm_results.csv",
    index=False,
)

print("\nMiniLM results saved successfully!")
print("Saved to:", results_dir / "minilm_results.csv")
print("Output columns:", output.columns.tolist())
print("Missing predictions:", output["minilm_similarity"].isna().sum())
print("Duplicate source row IDs:", output["source_row_id"].duplicated().sum())
