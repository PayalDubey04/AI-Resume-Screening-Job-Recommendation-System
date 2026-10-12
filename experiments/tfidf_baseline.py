
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
df = pd.read_csv(ROOT / "data" / "resume_data.csv")

# Step 2: Select relevant columns
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

required = resume_columns + job_columns + ["matched_score"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

# Step 3: Combine resume and job information
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

if len(df) < 3:
    raise ValueError("Not enough valid rows for evaluation.")

# Step 4: Split dataset
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42
)

# Step 5: Train TF-IDF vocabulary on training text only
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

training_text = pd.concat([
    train_df["resume_text"],
    train_df["job_text"]
])

vectorizer.fit(training_text)

# Step 6: Transform test resume and job text
resume_vectors = vectorizer.transform(test_df["resume_text"])
job_vectors = vectorizer.transform(test_df["job_text"])

# Step 7: Calculate pairwise cosine similarity
predictions = 1 - paired_cosine_distances(
    resume_vectors, job_vectors
)

predictions = np.clip(predictions, 0, 1)
actual = test_df["matched_score"].to_numpy()

# Step 8: Evaluate predictions
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

pearson = pearsonr(actual, predictions).statistic
spearman = spearmanr(actual, predictions).statistic

print("TF-IDF BASELINE RESULTS")
print("----------------------")
print("Test samples:", len(test_df))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("Pearson correlation:", round(pearson, 4))
print("Spearman correlation:", round(spearman, 4))

# Step 9: Save results
output = test_df[["matched_score"]].copy()
output["tfidf_similarity"] = predictions

results_dir = ROOT / "experiments" / "results"
results_dir.mkdir(parents=True, exist_ok=True)

output.to_csv(
    results_dir / "tfidf_results.csv",
    index=False
)

print("Results saved successfully!")
