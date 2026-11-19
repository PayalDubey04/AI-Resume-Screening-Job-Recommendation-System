import os
import re
import fitz
import numpy as np
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Field-Aware MPNet model once at startup
MODEL_NAME = "all-mpnet-base-v2"
print(f"Loading NLP model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully!")

WEIGHTS = {
    "skills": 0.50,
    "education": 0.20,
    "experience": 0.30,
}


def extract_pdf_text(file_storage) -> str:
    """Save uploaded PDF and extract plain text using PyMuPDF."""
    filename = os.path.basename(file_storage.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_storage.save(file_path)

    extracted = []
    with fitz.open(file_path) as doc:
        for page in doc:
            extracted.append(page.get_text())

    return "\n".join(extracted).strip()


def split_resume_sections(text: str) -> dict:
    """
    Split plain resume text into skills, education, and experience sections
    using common resume section headings. Falls back to full text if a heading
    is not explicitly detected.
    """
    sections = {"skills": "", "education": "", "experience": ""}
    current = None

    heading_patterns = {
        "skills": re.compile(r"^\s*(technical\s+)?(skills|core\s+competencies|technologies)\s*:?\s*$", re.I),
        "education": re.compile(r"^\s*(education|academic|qualifications)\s*:?\s*$", re.I),
        "experience": re.compile(r"^\s*(work\s+)?(experience|employment|projects|internships)\s*:?\s*$", re.I),
    }

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        matched_heading = None
        for sec_name, pattern in heading_patterns.items():
            if pattern.match(stripped):
                matched_heading = sec_name
                break

        if matched_heading:
            current = matched_heading
        elif current:
            sections[current] += " " + stripped

    # Fallback: if a section wasn't separated by headings, use full resume text
    for key in sections:
        sections[key] = sections[key].strip() if sections[key].strip() else text.strip()

    return sections


def compute_cosine_01(text_a: str, text_b: str):
    """Compute normalized MPNet cosine similarity scaled from [-1, 1] to [0, 1]."""
    text_a = (text_a or "").strip()
    text_b = (text_b or "").strip()
    if not text_a or not text_b:
        return None

    vecs = model.encode(
        [text_a, text_b],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    cos_sim = float(np.dot(vecs[0], vecs[1]))
    scaled = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))
    return round(scaled, 4)


def find_missing_skills(resume_text: str, required_skills_text: str) -> list:
    """
    Compare comma/newline-separated required skills against resume text
    (case-insensitive substring match). Only checks explicitly provided skills.
    """
    if not required_skills_text or not resume_text:
        return []

    resume_lower = resume_text.lower()
    raw_items = re.split(r"[,;\n•|]+", required_skills_text)
    missing = []

    for item in raw_items:
        skill = item.strip()
        if len(skill) >= 2 and skill.lower() not in resume_lower:
            missing.append(skill)

    return missing


def classify_match(score: float) -> str:
    """Categorize [0, 1] match score into Low, Medium, or High match."""
    if score >= 0.75:
        return "High Match"
    if score >= 0.55:
        return "Medium Match"
    return "Low Match"


def evaluate_match(resume_text: str, job_skills: str, job_education: str, job_experience: str) -> dict:
    """Run Field-Aware MPNet with dynamic weight renormalization."""
    resume_secs = split_resume_sections(resume_text)

    field_scores = {
        "skills": compute_cosine_01(resume_secs["skills"], job_skills),
        "education": compute_cosine_01(resume_secs["education"], job_education),
        "experience": compute_cosine_01(resume_secs["experience"], job_experience),
    }

    numerator = 0.0
    denominator = 0.0
    for field, score in field_scores.items():
        if score is not None:
            numerator += WEIGHTS[field] * score
            denominator += WEIGHTS[field]

    overall = round(numerator / denominator, 4) if denominator > 0 else 0.0
    missing_skills = find_missing_skills(resume_text, job_skills)

    return {
        "match_score": overall,
        "match_percentage": round(overall * 100, 1),
        "match_category": classify_match(overall),
        "skills_score": field_scores["skills"],
        "education_score": field_scores["education"],
        "experience_score": field_scores["experience"],
        "missing_skills": missing_skills,
    }


@app.route("/", methods=["GET", "POST"])
def home():
    extracted_text = ""
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("resume")
        job_skills = request.form.get("job_skills", "").strip()
        job_education = request.form.get("job_education", "").strip()
        job_experience = request.form.get("job_experience", "").strip()

        if not file or not file.filename.lower().endswith(".pdf"):
            error = "Please upload a valid PDF resume."
        else:
            extracted_text = extract_pdf_text(file)
            if not extracted_text:
                error = "Could not extract text from the uploaded PDF."
            elif not any([job_skills, job_education, job_experience]):
                error = "Please enter at least one job requirement field."
            else:
                result = evaluate_match(
                    extracted_text, job_skills, job_education, job_experience
                )

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        result=result,
        error=error,
    )


@app.route("/api/match", methods=["POST"])
def api_match():
    """JSON endpoint for frontend integration."""
    resume_text = ""

    if "resume" in request.files and request.files["resume"].filename:
        file = request.files["resume"]
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported."}), 400
        resume_text = extract_pdf_text(file)
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        resume_text = str(data.get("resume_text", "")).strip()
    else:
        resume_text = request.form.get("resume_text", "").strip()

    if not resume_text:
        return jsonify({"error": "Resume PDF or resume_text is required."}), 400

    if request.is_json:
        data = request.get_json(silent=True) or {}
        job_skills = str(data.get("job_skills", "")).strip()
        job_education = str(data.get("job_education", "")).strip()
        job_experience = str(data.get("job_experience", "")).strip()
    else:
        job_skills = request.form.get("job_skills", "").strip()
        job_education = request.form.get("job_education", "").strip()
        job_experience = request.form.get("job_experience", "").strip()

    if not any([job_skills, job_education, job_experience]):
        return jsonify({"error": "Provide at least one of: job_skills, job_education, job_experience."}), 400

    result = evaluate_match(resume_text, job_skills, job_education, job_experience)
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)