import os
import re
import fitz
import numpy as np
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Field-Aware MPNet semantic model once at startup
MODEL_NAME = "all-mpnet-base-v2"
print(f"Loading semantic model: {MODEL_NAME}...")
mpnet_model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully!")

SKILLS = [
    "python", "java", "c++", "sql", "excel", "power bi",
    "tableau", "machine learning", "deep learning",
    "data analysis", "data analytics", "statistics",
    "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "nlp", "html", "css", "javascript",
    "flask", "django", "git", "github", "communication",
    "leadership", "problem solving", "project management",
    "aws", "azure", "docker", "linux", "mongodb", "mysql",
    "postgresql", "etl", "data modeling", "data warehousing",
    "data integration", "apache spark", "data structures",
    "algorithms", "oop", "rest api", "rest apis"
]

SKILL_ALIASES = {
    "natural language processing": "nlp",
    "ml": "machine learning",
    "dl": "deep learning",
    "object-oriented programming": "oop",
    "object oriented programming": "oop",
    "relational dbms": "sql",
    "dbms": "sql",
    "etl & data pipelines": "etl",
    "data pipelines": "etl",
    "spark": "apache spark",
    "sklearn": "scikit-learn",
}

JOB_CATALOG = [
    {
        "title": "Data Engineering Intern",
        "company": "Honeywell / Enterprise Data Systems",
        "skills": ["python", "sql", "etl", "data modeling", "data warehousing", "data integration", "mysql", "git"],
        "description": "Design reliable ETL pipelines, build normalized relational data models, and support enterprise data warehousing workflows.",
    },
    {
        "title": "Machine Learning & NLP Intern",
        "company": "AetherAI Labs",
        "skills": ["python", "machine learning", "nlp", "pandas", "numpy", "scikit-learn", "git"],
        "description": "Develop and evaluate NLP text-matching pipelines, semantic embedding models, and preprocessing workflows in Python.",
    },
    {
        "title": "Backend Software Developer Intern",
        "company": "CloudScale Systems",
        "skills": ["python", "flask", "rest apis", "sql", "mysql", "git", "docker", "oop"],
        "description": "Build scalable Python/Flask web services, design relational database schemas, and integrate REST APIs with frontend apps.",
    },
    {
        "title": "Data Analyst & Research Intern",
        "company": "InsightGrid Analytics",
        "skills": ["python", "sql", "pandas", "statistics", "excel", "tableau", "data analysis"],
        "description": "Analyze structured datasets, run statistical comparisons, and deliver clear analytical insights to product teams.",
    },
    {
        "title": "Java & Full-Stack Developer Intern",
        "company": "NexaCore Technologies",
        "skills": ["java", "sql", "oop", "data structures", "algorithms", "html", "css", "git"],
        "description": "Develop object-oriented Java backend modules, manage MySQL databases, and build responsive web interfaces.",
    },
]


def extract_text_from_file(file_storage, file_path: str) -> str:
    """Extract plain text from uploaded PDF or TXT file."""
    file_storage.save(file_path)
    if file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    extracted = []
    with fitz.open(file_path) as document:
        for page in document:
            extracted.append(page.get_text())
    return "\n".join(extracted).strip()


def extract_contact_info(text: str) -> dict:
    """Extract emails and phone numbers via regex."""
    emails = sorted(set(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)))
    raw_phones = re.findall(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,5}\)?[\s-]?)?\d{3,5}[\s-]?\d{4,5}", text)
    phones = []
    for p in raw_phones:
        digits = re.sub(r"\D", "", p)
        if 10 <= len(digits) <= 13 and p.strip() not in phones:
            phones.append(p.strip())
    return {"emails": emails, "phones": phones[:2]}


def extract_sections(text: str) -> dict:
    """Detect standard resume sections for structured display and field-aware scoring."""
    sections = {}
    current_section = None

    heading_patterns = {
        "professional_summary": re.compile(r"^\s*(professional\s+)?(summary|objective|profile)\s*:?\s*$", re.I),
        "education": re.compile(r"^\s*(education|academic|qualifications)\s*:?\s*$", re.I),
        "skills": re.compile(r"^\s*(technical\s+)?(skills|core\s+competencies|technologies)\s*:?\s*$", re.I),
        "experience_and_projects": re.compile(r"^\s*(work\s+experience|experience|internships|academic\s+projects|projects)\s*:?\s*$", re.I),
        "coursework_and_certifications": re.compile(r"^\s*(relevant\s+coursework|coursework|certifications|achievements)\s*:?\s*$", re.I),
    }

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        matched_key = None
        for sec_key, pattern in heading_patterns.items():
            if pattern.match(stripped):
                matched_key = sec_key
                break

        if matched_key:
            current_section = matched_key
            sections.setdefault(current_section, "")
        elif current_section:
            sections[current_section] = (sections[current_section] + " " + stripped).strip()

    return {k: v for k, v in sections.items() if v}


def normalize_text_aliases(text: str) -> str:
    lowered = text.lower()
    for alias, canonical in SKILL_ALIASES.items():
        lowered = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", canonical, lowered)
    return lowered


def extract_skills(text: str) -> list:
    norm_text = normalize_text_aliases(text)
    found_skills = []
    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, norm_text):
            found_skills.append(skill)
    return sorted(set(found_skills))


def compute_tfidf_similarity(text_a: str, text_b: str) -> float:
    """Compute lexical TF-IDF cosine similarity percentage [0, 100]."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
    sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    return round(max(0.0, min(100.0, sim * 100.0)), 1)


def compute_mpnet_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic MPNet cosine similarity scaled to percentage [0, 100]."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vecs = mpnet_model.encode([text_a, text_b], normalize_embeddings=True, show_progress_bar=False)
    cos_sim = float(np.dot(vecs[0], vecs[1]))
    scaled = ((cos_sim + 1.0) / 2.0) * 100.0
    return round(max(0.0, min(100.0, scaled)), 1)


def recommend_jobs(resume_text: str, resume_skills: list, top_k: int = 3) -> list:
    """Hybrid Top-K job recommender combining skill overlap (55%) + MPNet semantic fit (45%)."""
    recommendations = []
    resume_set = set(resume_skills)

    for job in JOB_CATALOG:
        req_skills = job["skills"]
        matched = [s for s in req_skills if s in resume_set]
        missing = [s for s in req_skills if s not in resume_set]

        skill_pct = (len(matched) / len(req_skills) * 100.0) if req_skills else 0.0
        role_text = f"{job['title']}. {job['description']} Required skills: {', '.join(req_skills)}"
        semantic_pct = compute_mpnet_similarity(resume_text, role_text)

        hybrid_score = round((0.55 * skill_pct) + (0.45 * semantic_pct), 1)

        recommendations.append({
            "title": job["title"],
            "company": job["company"],
            "description": job["description"],
            "score": hybrid_score,
            "matched_skills": matched,
            "missing_skills": missing,
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:top_k]


@app.route("/", methods=["GET", "POST"])
def home():
    extracted_text = ""
    job_description = ""
    analysis = None
    recommendations = []
    error = None

    if request.method == "POST":
        file = request.files.get("resume")
        job_description = request.form.get("job_description", "").strip()

        if not file or not file.filename:
            error = "Please upload a resume (PDF or TXT)."
        elif not file.filename.lower().endswith((".pdf", ".txt")):
            error = "Only PDF or TXT files are supported."
        elif not job_description:
            error = "Please paste a job description."
        else:
            safe_filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_filename)

            try:
                extracted_text = extract_text_from_file(file, file_path)
                if not extracted_text:
                    error = "Could not extract text from the uploaded file."
                else:
                    contact_info = extract_contact_info(extracted_text)
                    sections = extract_sections(extracted_text)

                    resume_skills = extract_skills(extracted_text)
                    jd_skills = extract_skills(job_description)

                    matched_skills = [s for s in jd_skills if s in resume_skills]
                    missing_skills = [s for s in jd_skills if s not in resume_skills]
                    extra_skills = [s for s in resume_skills if s not in jd_skills]

                    # Hybrid compatibility: combines skill overlap + Field-Aware MPNet semantic match
                    skill_overlap_pct = (len(matched_skills) / len(jd_skills) * 100.0) if jd_skills else 0.0
                    semantic_pct = compute_mpnet_similarity(extracted_text, job_description)
                    tfidf_pct = compute_tfidf_similarity(extracted_text, job_description)

                    if jd_skills:
                        overall_score = round((0.60 * skill_overlap_pct) + (0.40 * semantic_pct), 1)
                        missing_pct = round((len(missing_skills) / len(jd_skills)) * 100.0, 1)
                    else:
                        overall_score = semantic_pct
                        missing_pct = 0.0

                    analysis = {
                        "contact_info": contact_info,
                        "sections": sections,
                        "required_skills": jd_skills,
                        "matched_skills": matched_skills,
                        "missing_skills": missing_skills,
                        "extra_skills": extra_skills,
                        "match_score": overall_score,
                        "missing_percentage": missing_pct,
                        "text_similarity": tfidf_pct,
                    }

                    recommendations = recommend_jobs(extracted_text, resume_skills, top_k=3)

            except Exception as e:
                error = f"Error processing resume: {e}"

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        job_description=job_description,
        analysis=analysis,
        recommendations=recommendations,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)