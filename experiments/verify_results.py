
from pathlib import Path
import math
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

files = {
    "TF-IDF": "tfidf_results.csv",
    "MiniLM": "minilm_results.csv",
    "MPNet": "mpnet_results.csv",
    "Field-Aware MPNet": "field_aware_mpnet_results.csv",
}

prediction_columns = {
    "TF-IDF": "tfidf_similarity",
    "MiniLM": "minilm_similarity",
    "MPNet": "mpnet_similarity",
    "Field-Aware MPNet": "field_aware_score",
}

datasets = {}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def normalize_ids(series):
    """Normalize IDs for comparison without changing their order."""
    return series.reset_index(drop=True)


# ============================================================
# 1. LOAD AND INSPECT RESULT FILES
# ============================================================

section("RESUME SCREENING MODEL VERIFICATION")

for model, filename in files.items():
    path = RESULTS / filename

    if not path.exists():
        print(f"\n{model}: FILE NOT FOUND - {path}")
        continue

    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
    except Exception as exc:
        print(f"\n{model}: ERROR READING FILE - {exc}")
        continue

    datasets[model] = df

    print(f"\n{model}")
    print("-" * 45)
    print("File:", filename)
    print("Test samples:", len(df))
    print("Columns:", df.columns.tolist())
    print("Total missing values:", int(df.isna().sum().sum()))

    # Inspect missing values by column
    missing_by_column = df.isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0]

    if not missing_by_column.empty:
        print("Missing values by column:")
        print(missing_by_column.to_string())
    else:
        print("Missing values by column: None")

    # Check target scores
    if "matched_score" not in df.columns:
        print("WARNING: matched_score column is missing.")
    else:
        scores = pd.to_numeric(df["matched_score"], errors="coerce")
        print("Invalid/missing target scores:", int(scores.isna().sum()))

    # Check prediction column
    pred_col = prediction_columns[model]

    if pred_col not in df.columns:
        print(f"WARNING: Prediction column '{pred_col}' is missing.")
    else:
        predictions = pd.to_numeric(df[pred_col], errors="coerce")
        print("Prediction column:", pred_col)
        print("Missing/invalid predictions:", int(predictions.isna().sum()))

        outside_range = (
            predictions.notna()
            & ~predictions.between(0, 1)
        )
        print("Predictions outside [0, 1]:", int(outside_range.sum()))

    # Check row IDs
    if "source_row_id" in df.columns:
        ids = df["source_row_id"]
        print("Missing source row IDs:", int(ids.isna().sum()))
        print("Duplicate source row IDs:", int(ids.duplicated().sum()))
    else:
        print("WARNING: source_row_id is missing.")


# ============================================================
# 2. FILE COMPLETENESS
# ============================================================

section("FILE COMPLETENESS")

missing_models = [model for model in files if model not in datasets]

if missing_models:
    print("Missing or unreadable models:", missing_models)
else:
    print("All four model result files were loaded successfully.")


# ============================================================
# 3. TEST ROW ALIGNMENT
# ============================================================

section("TEST ROW ALIGNMENT")

if len(datasets) >= 2:
    reference_model = next(iter(datasets))
    reference_df = datasets[reference_model]

    if "source_row_id" not in reference_df.columns:
        print(
            f"Cannot verify row alignment: {reference_model} "
            "has no source_row_id column."
        )
    else:
        reference_ids = normalize_ids(reference_df["source_row_id"])

        for model, df in datasets.items():
            if model == reference_model:
                continue

            if "source_row_id" not in df.columns:
                print(f"{reference_model} vs {model}: ID column missing.")
                continue

            other_ids = normalize_ids(df["source_row_id"])

            if len(reference_ids) != len(other_ids):
                print(f"{reference_model} vs {model}: Different row counts.")
                print("  Reference rows:", len(reference_ids))
                print("  Other rows:", len(other_ids))
                continue

            same_order = reference_ids.equals(other_ids)

            # Compare ID sets while preserving duplicate counts.
            ref_valid = reference_ids.dropna()
            other_valid = other_ids.dropna()

            same_records = (
                len(ref_valid) == len(other_valid)
                and ref_valid.value_counts().sort_index().equals(
                    other_valid.value_counts().sort_index()
                )
                and reference_ids.isna().sum() == other_ids.isna().sum()
            )

            print(f"\n{reference_model} vs {model}:")
            print("  Same test rows in same order:", same_order)
            print("  Same set of test rows:", same_records)
else:
    print("At least two result files are needed for comparison.")


# ============================================================
# 4. TARGET SCORE CONSISTENCY
# ============================================================

section("TARGET SCORE SEQUENCE CHECK")

if len(datasets) >= 2:
    reference_model = next(iter(datasets))
    reference_df = datasets[reference_model]

    if "matched_score" not in reference_df.columns:
        print(f"{reference_model}: matched_score column is missing.")
    else:
        reference_scores = pd.to_numeric(
            reference_df["matched_score"], errors="coerce"
        ).reset_index(drop=True)

        for model, df in datasets.items():
            if model == reference_model:
                continue

            if "matched_score" not in df.columns:
                print(f"{model}: matched_score column is missing.")
                continue

            other_scores = pd.to_numeric(
                df["matched_score"], errors="coerce"
            ).reset_index(drop=True)

            same = (
                len(reference_scores) == len(other_scores)
                and reference_scores.equals(other_scores)
            )

            print(f"{reference_model} vs {model}: {same}")
else:
    print("Not enough result files to compare.")


# ============================================================
# 5. PREDICTION COLUMN CHECK
# ============================================================

section("PREDICTION COLUMN CHECK")

for model, df in datasets.items():
    expected = prediction_columns[model]

    if expected in df.columns:
        print(f"{model}: OK - {expected}")
    else:
        print(f"{model}: MISSING - {expected}")


# ============================================================
# 6. MODEL METRICS COMPARISON
# ============================================================

section("MODEL METRICS COMPARISON")

metrics = []

for model, df in datasets.items():
    pred_col = prediction_columns[model]

    if "matched_score" not in df.columns:
        print(f"{model}: Cannot calculate metrics - target missing.")
        continue

    if pred_col not in df.columns:
        print(f"{model}: Cannot calculate metrics - prediction missing.")
        continue

    actual = pd.to_numeric(df["matched_score"], errors="coerce")
    predicted = pd.to_numeric(df[pred_col], errors="coerce")

    # Use valid numeric pairs. Keep this mask consistent with
    # the intended evaluation range of the project.
    valid = (
        actual.notna()
        & predicted.notna()
        & actual.between(0, 1)
        & predicted.between(0, 1)
    )

    if valid.sum() == 0:
        print(f"{model}: No valid prediction-score pairs.")
        continue

    y_true = actual[valid]
    y_pred = predicted[valid]

    mae = (y_true - y_pred).abs().mean()
    rmse = math.sqrt(((y_true - y_pred) ** 2).mean())

    if y_true.nunique() > 1 and y_pred.nunique() > 1:
        pearson = y_true.corr(y_pred, method="pearson")
        spearman = y_true.corr(y_pred, method="spearman")
    else:
        pearson = float("nan")
        spearman = float("nan")

    metrics.append({
        "Model": model,
        "Samples": int(valid.sum()),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Pearson": round(pearson, 4),
        "Spearman": round(spearman, 4),
    })

if metrics:
    metrics_df = pd.DataFrame(metrics)

    print(metrics_df.to_string(index=False))

    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS / "model_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\nMetrics saved to:", metrics_path)
else:
    print("No model metrics could be calculated.")


# ============================================================
# 7. FINAL STATUS
# ============================================================

section("VERIFICATION FINISHED")

if missing_models:
    print("WARNING: One or more result files are missing or unreadable.")
else:
    print("All expected result files were loaded.")

print(
    "\nImportant:"
    "\n- Matching target scores alone does not prove identical resume-job pairs."
    "\n- Use the same test records in the same order for fair comparison."
    "\n- Check the missing-value report, especially for Field-Aware MPNet."
    "\n- Metrics above use numeric target and prediction values in [0, 1]."
)
