
from pathlib import Path
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

# 1. Load dataset
ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "resume_data.csv")

# 2. Select resume and job columns
resume_columns = [
    "career_objective",
    "skills",
    "degree_names",
    "major_field_of_studies",
    "positions",
    "responsibilities"
]

job_columns = [
    "role_positions",
    "educationaL_requirements",
    "experiencere_requirement",
    "skills_required",
    "responsibilities.1"
]

# Remove whitespace from column names
df.columns = df.columns.str.strip()

# Check for missing columns
required = resume_columns + job_columns + ["matched_score"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# 3. Combine text
def combine_text(frame, columns):
    return frame[columns].fillna("").astype(str).agg(" ".join, axis=1)

df["resume_text"] = combine_text(df, resume_columns)
df["job_text"] = combine_text(df, job_columns)

df["matched_score"] = pd.to_numeric(
    df["matched_score"], errors="coerce"
)

df = df.dropna(subset=["matched_score"]).copy()

df = df[
    df["resume_text"].str.strip().ne("") &
    df["job_text"].str.strip().ne("")
].copy()

# 4. Use same split as TF-IDF
_, test_df = train_test_split(
    df, test_size=0.2, random_state=42
)

# 5. Load MiniLM
print("Loading MiniLM model...")

model = SentenceTransformer("all-mpnet-base-v2")

# 6. Generate embeddings
print("Encoding resumes...")
resume_embeddings = model.encode(
    test_df["resume_text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Encoding job descriptions...")
job_embeddings = model.encode(
    test_df["job_text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

# 7. Calculate pairwise similarity
similarities = np.sum(
    resume_embeddings * job_embeddings, axis=1
)

# Convert cosine similarity to 0-1 scale
predictions = np.clip((similarities + 1) / 2, 0, 1)

# 8. Evaluate
actual = test_df["matched_score"].to_numpy()

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

pearson = pearsonr(actual, predictions).statistic
spearman = spearmanr(actual, predictions).statistic

print("\nMPNET RESULTS")
print("--------------------")
print("Test samples:", len(test_df))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("Pearson:", round(pearson, 4))
print("Spearman:", round(spearman, 4))

# 9. Save results
results_dir = ROOT / "experiments" / "results"
results_dir.mkdir(parents=True, exist_ok=True)

output = test_df[["matched_score"]].copy()
output["mpnet_similarity"] = predictions

output.to_csv(
    results_dir / "mpnet_results.csv",
    index=False
)

print("MiniLM results saved successfully!")
