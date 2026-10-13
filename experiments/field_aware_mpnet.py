
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

MODEL_NAME = "all-mpnet-base-v2"
RANDOM_STATE = 42
TEST_SIZE = 0.20
BATCH_SIZE = 32


# ============================================================
# 2. LOAD DATA AND NORMALIZE COLUMN NAMES
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")


def clean_column_name(col):
    """Normalize column names and remove surrounding quotes."""
    col = str(col).replace("\ufeff", "").strip()

    while len(col) >= 2 and (
        (col.startswith('"') and col.endswith('"'))
        or (col.startswith("'") and col.endswith("'"))
    ):
        col = col[1:-1].strip()

    return col


df.columns = [clean_column_name(col) for col in df.columns]

column_lookup = {}
for col in df.columns:
    column_lookup.setdefault(col.strip().lower(), col)


def resolve_column(candidates, required=False):
    """Return the actual column name matching a candidate."""
    for candidate in candidates:
        actual = column_lookup.get(candidate.strip().lower())
        if actual is not None:
            return actual

    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return None


education_job_col = resolve_column([
    "educational_requirements",
    "educational_requirement",
    "education_requirements",
    "education_requirement",
    "education",
])

experience_job_col = resolve_column([
    "experiencere_requirement",
    "experience_requirement",
    "experience_requirements",
    "experiencerequirement",
    "experience_required",
])

matched_score_col = resolve_column(["matched_score"], required=True)
skills_col = resolve_column(["skills"], required=True)
skills_required_col = resolve_column(["skills_required"], required=True)

degree_col = resolve_column(["degree_names"])
major_col = resolve_column(["major_field_of_studies"])
positions_col = resolve_column(["positions"])
responsibilities_col = resolve_column(["responsibilities"])
responsibilities_job_col = resolve_column(["responsibilities.1"])

# Preserve the original dataset row identity.
df["source_row_id"] = np.arange(len(df), dtype=int)

print("=" * 65)
print("FIELD-AWARE MPNet BASELINE")
print("=" * 65)
print("Dataset shape:", df.shape)
print("Education requirements column:", education_job_col)
print("Experience requirements column:", experience_job_col)


# ============================================================
# 3. DEFINE FIELD-AWARE MATCHING COMPONENTS
# ============================================================

fields = {
    "skills": {
        "resume": [skills_col],
        "job": [skills_required_col],
        "weight": 0.50,
    },
    "education": {
        "resume": [degree_col, major_col],
        "job": [education_job_col],
        "weight": 0.20,
    },
    "experience": {
        "resume": [positions_col, responsibilities_col],
        "job": [experience_job_col, responsibilities_job_col],
        "weight": 0.30,
    },
}

# Remove columns that do not exist in the dataset.
for spec in fields.values():
    spec["resume"] = [
        col for col in spec["resume"] if col is not None
    ]
    spec["job"] = [
        col for col in spec["job"] if col is not None
    ]

if not fields["skills"]["resume"] or not fields["skills"]["job"]:
    raise ValueError(
        "Required skills columns are missing. "
        f"Available columns: {df.columns.tolist()}"
    )

usable_fields = {
    name: spec
    for name, spec in fields.items()
    if spec["resume"] and spec["job"]
}

if not usable_fields:
    raise ValueError("No usable matching components were found.")


# ============================================================
# 4. PREPARE RESUME AND JOB TEXT
# ============================================================

def combine_text(frame, columns):
    """Combine existing columns into clean text strings."""
    if not columns:
        return pd.Series("", index=frame.index, dtype="object")

    return (
        frame[columns]
        .fillna("")
        .astype(str)
        .apply(lambda col: col.str.strip())
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


resume_cols = list(dict.fromkeys(
    col
    for spec in usable_fields.values()
    for col in spec["resume"]
))

job_cols = list(dict.fromkeys(
    col
    for spec in usable_fields.values()
    for col in spec["job"]
))

df["resume_text"] = combine_text(df, resume_cols)
df["job_text"] = combine_text(df, job_cols)

df["matched_score"] = pd.to_numeric(
    df[matched_score_col], errors="coerce"
)

before_filter = len(df)

df = df.dropna(subset=["matched_score"]).copy()
df = df[
    df["resume_text"].ne("") & df["job_text"].ne("")
].copy()

print("Rows before filtering:", before_filter)
print("Valid records:", len(df))
print("Rows removed:", before_filter - len(df))
print("Components used:", list(usable_fields.keys()))

if len(df) < 3:
    raise ValueError(
        f"Not enough valid records to evaluate: {len(df)}"
    )

if (~df["matched_score"].between(0, 1)).any():
    print(
        "WARNING: Some matched_score values are outside [0, 1]. "
        "Check the target scale before interpreting metrics."
    )


# ============================================================
# 5. TRAIN/TEST SPLIT
# ============================================================

_, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

test_df = test_df.copy()
print("Test samples:", len(test_df))


# ============================================================
# 6. LOAD MPNet
# ============================================================

print(f"\nLoading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)


# ============================================================
# 7. FIELD-LEVEL SEMANTIC SIMILARITY
# ============================================================

def field_similarity(resume_series, job_series, field_name):
    """
    Calculate similarity for non-empty field pairs.

    Missing field pairs remain NaN. The final score uses available
    components and renormalizes their weights for each row.
    """
    resume_texts = (
        resume_series.fillna("").astype(str).str.strip().tolist()
    )
    job_texts = (
        job_series.fillna("").astype(str).str.strip().tolist()
    )

    scores = np.full(len(resume_texts), np.nan, dtype=float)

    valid_indices = [
        i
        for i, (resume, job) in enumerate(
            zip(resume_texts, job_texts)
        )
        if resume and job
    ]

    print(
        f"  {field_name}: {len(valid_indices)} valid pairs; "
        f"{len(scores) - len(valid_indices)} empty/missing pairs"
    )

    if not valid_indices:
        return scores

    resume_vectors = model.encode(
        [resume_texts[i] for i in valid_indices],
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    job_vectors = model.encode(
        [job_texts[i] for i in valid_indices],
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    # Normalized embeddings allow dot product to calculate cosine similarity.
    cosine_scores = np.sum(resume_vectors * job_vectors, axis=1)

    # Map cosine similarity from [-1, 1] to [0, 1].
    scaled_scores = np.clip(
        (cosine_scores + 1.0) / 2.0, 0.0, 1.0
    )

    scores[valid_indices] = scaled_scores
    return scores


for name, spec in usable_fields.items():
    resume_text = combine_text(test_df, spec["resume"])
    job_text = combine_text(test_df, spec["job"])

    print(f"Calculating {name} similarity...")

    test_df[f"{name}_similarity"] = field_similarity(
        resume_text,
        job_text,
        name,
    )


# ============================================================
# 8. CALCULATE WEIGHTED FIELD-AWARE SCORE
# ============================================================

component_columns = {
    name: f"{name}_similarity"
    for name in usable_fields
}

numerator = np.zeros(len(test_df), dtype=float)
denominator = np.zeros(len(test_df), dtype=float)

for name, column in component_columns.items():
    scores = test_df[column].to_numpy(dtype=float)
    weight = float(usable_fields[name]["weight"])

    available = np.isfinite(scores)

    numerator[available] += weight * scores[available]
    denominator[available] += weight

# Renormalize weights over components available for each row.
predictions = np.divide(
    numerator,
    denominator,
    out=np.full(len(test_df), np.nan, dtype=float),
    where=denominator > 0,
)

test_df["field_aware_score"] = np.clip(
    predictions, 0.0, 1.0
)

print("\nMissing component scores:")
for name, column in component_columns.items():
    print(
        f"  {column}: "
        f"{int(test_df[column].isna().sum())}"
    )

print(
    "Missing overall predictions:",
    int(test_df["field_aware_score"].isna().sum()),
)


# ============================================================
# 9. EVALUATE RESULTS
# ============================================================

valid = (
    test_df["matched_score"].notna()
    & test_df["field_aware_score"].notna()
)

actual = test_df.loc[
    valid, "matched_score"
].to_numpy(dtype=float)

pred = test_df.loc[
    valid, "field_aware_score"
].to_numpy(dtype=float)

if len(actual) == 0:
    raise ValueError("No valid predictions available for evaluation.")

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))

if (
    len(actual) >= 2
    and np.std(actual) > 0
    and np.std(pred) > 0
):
    pearson = pearsonr(actual, pred).statistic
    spearman = spearmanr(actual, pred).statistic
else:
    pearson = np.nan
    spearman = np.nan

print("\nFIELD-AWARE MPNet RESULTS")
print("-------------------------")
print("Evaluated samples:", len(actual))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print(
    "Pearson:",
    round(pearson, 4) if np.isfinite(pearson) else "undefined",
)
print(
    "Spearman:",
    round(spearman, 4) if np.isfinite(spearman) else "undefined",
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

output_columns = [
    "source_row_id",
    "matched_score",
    "field_aware_score",
]

for name in fields:
    column = f"{name}_similarity"
    if column in test_df.columns:
        output_columns.append(column)

output_path = RESULTS_DIR / "field_aware_mpnet_results.csv"

test_df[output_columns].to_csv(
    output_path,
    index=False,
)

print("\nResults saved to:")
print(output_path)
print("Finished.")

print(
    "\nNote: Missing field similarities are kept blank when either "
    "side has no text. The overall score uses available components "
    "and renormalizes their weights."
)
