
from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "resume_data.csv"
RESULTS_DIR = ROOT / "experiments" / "results"
OUTPUT_PATH = RESULTS_DIR / "mpnet_results.csv"

MODEL_NAME = "all-mpnet-base-v2"
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32


# ============================================================
# 2. LOAD DATASET AND PRESERVE ORIGINAL ROW IDS
# ============================================================

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# Normalize column names
df.columns = (
    df.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.strip('"')
    .str.strip("'")
)

# Preserve the original CSV row identity before filtering
df["source_row_id"] = np.arange(len(df))

print("Dataset shape:", df.shape)

# Resume columns
resume_columns = [
    "career_objective",
    "skills",
    "degree_names",
    "major_field_of_studies",
    "positions",
    "responsibilities",
]

# Job columns
job_columns = [
    "role_positions",
    "educationaL_requirements",
    "experiencere_requirement",
    "skills_required",
    "responsibilities.1",
]

required_columns = (
    resume_columns
    + job_columns
    + ["matched_score"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )


# ============================================================
# 3. COMBINE RESUME AND JOB TEXT
# ============================================================

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
    df["matched_score"],
    errors="coerce",
)

# Apply the same filtering rules across all baselines
df = df.dropna(subset=["matched_score"]).copy()

df = df[
    df["resume_text"].str.strip().ne("")
    & df["job_text"].str.strip().ne("")
].copy()

if len(df) < 3:
    raise ValueError(
        "Not enough valid records to evaluate MPNet."
    )

print("Valid records:", len(df))


# ============================================================
# 4. CREATE THE TEST SPLIT
# ============================================================

_, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

test_df = test_df.copy()

print("MPNet test samples:", len(test_df))


# ============================================================
# 5. LOAD MPNet MODEL
# ============================================================

print("\nLoading MPNet model...")

model = SentenceTransformer(MODEL_NAME)


# ============================================================
# 6. GENERATE RESUME EMBEDDINGS
# ============================================================

print("\nEncoding resumes...")

resume_embeddings = model.encode(
    test_df["resume_text"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)


# ============================================================
# 7. GENERATE JOB DESCRIPTION EMBEDDINGS
# ============================================================

print("\nEncoding job descriptions...")

job_embeddings = model.encode(
    test_df["job_text"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)


# ============================================================
# 8. CALCULATE COSINE SIMILARITY
# ============================================================

# Embeddings are normalized, so their dot product is cosine similarity
similarities = np.sum(
    resume_embeddings * job_embeddings,
    axis=1,
)

# Map cosine similarity from [-1, 1] to [0, 1]
predictions = np.clip(
    (similarities + 1.0) / 2.0,
    0.0,
    1.0,
)


# ============================================================
# 9. EVALUATE MODEL
# ============================================================

actual = test_df["matched_score"].to_numpy(dtype=float)

valid = (
    np.isfinite(actual)
    & np.isfinite(predictions)
)

actual = actual[valid]
predictions = predictions[valid]
evaluated_df = test_df.iloc[np.flatnonzero(valid)].copy()

if len(actual) == 0:
    raise ValueError("No valid predictions available for evaluation.")

mae = mean_absolute_error(actual, predictions)

rmse = np.sqrt(
    mean_squared_error(actual, predictions)
)

if (
    len(actual) >= 2
    and np.std(actual) > 0
    and np.std(predictions) > 0
):
    pearson = pearsonr(actual, predictions).statistic
    spearman = spearmanr(actual, predictions).statistic
else:
    pearson = np.nan
    spearman = np.nan


print("\nMPNET RESULTS")
print("==============================")
print("Evaluated samples:", len(actual))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("Pearson:", round(pearson, 4))
print("Spearman:", round(spearman, 4))


# ============================================================
# 10. SAVE RESULTS WITH SOURCE ROW IDS
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

output = evaluated_df[
    ["source_row_id", "matched_score"]
].copy()

output["mpnet_similarity"] = predictions

output.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\nMPNet results saved successfully!")
print("Saved to:", OUTPUT_PATH)
print("Output columns:", output.columns.tolist())
print(
    "Missing predictions:",
    int(output["mpnet_similarity"].isna().sum()),
)
print(
    "Duplicate source row IDs:",
    int(output["source_row_id"].duplicated().sum()),
)
