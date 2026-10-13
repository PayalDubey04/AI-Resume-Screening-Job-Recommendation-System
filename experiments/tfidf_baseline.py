
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

# Step 1: Load dataset
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "resume_data.csv"

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# Preserve the original dataset row identity BEFORE filtering
df["source_row_id"] = np.arange(len(df))

# Step 2: Normalize column names
df.columns = (
    df.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.strip('"')
    .str.strip("'")
)

# Step 3: Select relevant columns
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
        f"Missing dataset columns: {missing}. "
        f"Available columns: {df.columns.tolist()}"
    )

# Step 4: Combine resume and job information
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

# Filter invalid records while preserving source_row_id
df = df.dropna(subset=["matched_score"]).copy()

df = df[
    df["resume_text"].ne("") &
    df["job_text"].ne("")
].copy()

if len(df) < 3:
    raise ValueError("Not enough valid rows for evaluation.")

# Step 5: Split dataset
# Keep the same settings as the other baseline scripts.
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
)

test_df = test_df.copy()

print("TF-IDF test samples:", len(test_df))

# Step 6: Train TF-IDF vocabulary on training text only
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
)

training_text = pd.concat(
    [
        train_df["resume_text"],
        train_df["job_text"],
    ],
    ignore_index=True,
)

vectorizer.fit(training_text)

# Step 7: Transform test resume and job text
resume_vectors = vectorizer.transform(test_df["resume_text"])
job_vectors = vectorizer.transform(test_df["job_text"])

# Step 8: Calculate cosine similarity
predictions = 1 - paired_cosine_distances(
    resume_vectors,
    job_vectors,
)

predictions = np.clip(predictions, 0, 1)
actual = test_df["matched_score"].to_numpy()

# Step 9: Evaluate predictions
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

print("\nTF-IDF BASELINE RESULTS")
print("-----------------------")
print("Test samples:", len(test_df))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("Pearson correlation:", round(pearson, 4))
print("Spearman correlation:", round(spearman, 4))

# Step 10: Save results with original row IDs
output = test_df[
    ["source_row_id", "matched_score"]
].copy()

output["tfidf_similarity"] = predictions

results_dir = ROOT / "experiments" / "results"
results_dir.mkdir(parents=True, exist_ok=True)

output.to_csv(
    results_dir / "tfidf_results.csv",
    index=False,
)

print("\nResults saved successfully!")
print("Saved to:", results_dir / "tfidf_results.csv")
print("Output columns:", output.columns.tolist())
print("Missing predictions:", output["tfidf_similarity"].isna().sum())
print("Duplicate source row IDs:", output["source_row_id"].duplicated().sum())
