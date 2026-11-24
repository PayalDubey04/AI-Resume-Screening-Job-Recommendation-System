import os
import re
import fitz

from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# ---------- SKILL DICTIONARY ----------

SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript",
    "html", "css", "react", "angular", "node.js",
    "flask", "django", "spring boot",
    "sql", "mysql", "postgresql", "mongodb",
    "machine learning", "deep learning", "nlp",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "git", "github", "docker", "kubernetes",
    "rest api", "linux", "aws", "azure",
    "data structures", "algorithms", "oop",
    "excel", "power bi", "tableau",
    "communication", "problem solving"
]
SKILL_ALIASES = {
    "javascript": ["js"],
    "typescript": ["ts"],
    "react": ["react.js", "reactjs"],
    "angular": ["angularjs", "angular.js"],
    "node.js": ["nodejs", "node"],
    "postgresql": ["postgres"],
    "machine learning": ["ml"],
    "deep learning": ["dl"],
    "nlp": ["natural language processing"],
    "scikit-learn": ["sklearn"],
    "rest api": ["rest apis", "restful api", "restful apis"],
    "oop": ["object oriented programming", "object-oriented programming"],
    "data structures": ["dsa", "data structures and algorithms"],
    "power bi": ["powerbi"]
}


# ---------- TEXT EXTRACTION ----------

def extract_text_from_file(file_path):

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        text = ""

        with fitz.open(file_path) as document:
            for page in document:
                text += page.get_text() + "\n"

        return text

    elif extension == ".txt":
        with open(
            file_path, "r", encoding="utf-8", errors="ignore"
        ) as file:
            return file.read()

    return ""


# ---------- SKILL DETECTION ----------

def extract_skills(text):

    cleaned_text = re.sub(r"\s+", " ", text.lower())
    found_skills = []

    for skill in SKILLS:
        variants = [skill] + SKILL_ALIASES.get(skill, [])

        for variant in variants:
            pattern = r"(?<![a-zA-Z0-9+#.])" + re.escape(variant) + r"(?![a-zA-Z0-9+#.])"
            if re.search(pattern, cleaned_text):
                found_skills.append(skill)
                break

    return sorted(set(found_skills))

    # ---------- CONTACT EXTRACTION ----------

def extract_contact_info(text):
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    raw_emails = re.findall(email_pattern, text)
    emails = list(dict.fromkeys(email.strip() for email in raw_emails))

    phone_pattern = r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{4}"
    raw_phones = re.findall(phone_pattern, text)
    phones = []
    for candidate in raw_phones:
        cleaned = candidate.strip()
        digits_only = re.sub(r"\D", "", cleaned)
        if 7 <= len(digits_only) <= 15 and cleaned not in phones:
            phones.append(cleaned)

    return {
        "emails": emails,
        "phones": phones
    }


    # ---------- SECTION DETECTION ----------

SECTION_HEADINGS = {
    "summary": {"summary", "professional summary", "career objective", "objective", "profile"},
    "skills": {"skills", "technical skills", "core skills", "key skills"},
    "education": {"education", "academic background", "academics", "qualifications"},
    "experience": {"experience", "work experience", "employment", "internships", "internship"},
    "projects": {"projects", "academic projects", "personal projects", "key projects"},
    "certifications": {"certifications", "certification", "courses", "training", "licenses"}
}


def extract_resume_sections(text):

    sections = {key: [] for key in SECTION_HEADINGS}
    current_section = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        normalized = re.sub(r"[:\-–—]+$", "", line).strip().lower()

        if len(normalized) <= 40:
            matched_key = None
            for key, aliases in SECTION_HEADINGS.items():
                if normalized in aliases:
                    matched_key = key
                    break

            if matched_key:
                current_section = matched_key
                continue

        if current_section:
            sections[current_section].append(line)

    return {
        key: "\n".join(lines)
        for key, lines in sections.items()
        if lines
    }


    # ---------- JOB CATALOGUE ----------

JOB_CATALOG = [
    {
        "title": "Python Developer",
        "company": "Demo Company A",
        "description": "Python backend development",
        "skills": [
            "python", "flask", "sql", "git",
            "rest api", "oop", "data structures"
        ]
    },
    {
        "title": "Java Developer",
        "company": "Demo Company B",
        "description": "Java application development",
        "skills": [
            "java", "spring boot", "sql", "git",
            "oop", "data structures", "algorithms"
        ]
    },
    {
        "title": "Data Analyst",
        "company": "Demo Company C",
        "description": "Data analysis and reporting",
        "skills": [
            "python", "sql", "excel", "pandas",
            "numpy", "power bi", "tableau"
        ]
    },
    {
        "title": "Machine Learning Engineer",
        "company": "Demo Company D",
        "description": "Machine learning model development",
        "skills": [
            "python", "machine learning", "pandas",
            "numpy", "scikit-learn", "deep learning"
        ]
    },
    {
        "title": "Frontend Developer",
        "company": "Demo Company E",
        "description": "Frontend web development",
        "skills": [
            "html", "css", "javascript", "react",
            "typescript", "git"
        ]
    },
    {
        "title": "NLP Engineer",
        "company": "Demo Company F",
        "description": "Natural language processing",
        "skills": [
            "python", "nlp", "machine learning",
            "pandas", "numpy", "scikit-learn"
        ]
    },
    {
        "title": "Full Stack Developer",
        "company": "Demo Company G",
        "description": "Frontend and backend development",
        "skills": [
            "html", "css", "javascript", "react",
            "python", "flask", "sql", "git", "rest api"
        ]
    }
]


# ---------- RESUME ANALYSIS ----------

def analyze_resume(resume_text, job_description):

    resume_skills = extract_skills(resume_text)
    required_skills = extract_skills(job_description)

    matched_skills = [
        skill for skill in required_skills
        if skill in resume_skills
    ]

    missing_skills = [
        skill for skill in required_skills
        if skill not in resume_skills
    ]

    extra_skills = [
        skill for skill in resume_skills
        if skill not in required_skills
    ]

    total_required = len(required_skills)

    if total_required:
        match_score = round(
            len(matched_skills) / total_required * 100
        )
    else:
        match_score = 0

    missing_percentage = 100 - match_score

    return {
        "contact_info": extract_contact_info(resume_text),
        "sections": extract_resume_sections(resume_text),
        "resume_skills": resume_skills,
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills": extra_skills,
        "match_score": match_score,
        "missing_percentage": missing_percentage
    }



    # ---------- JOB RECOMMENDATION ENGINE ----------

def recommend_jobs(resume_skills):

    recommendations = []

    for job in JOB_CATALOG:

        required_skills = job["skills"]

        matched = [
            skill for skill in required_skills
            if skill in resume_skills
        ]

        missing = [
            skill for skill in required_skills
            if skill not in resume_skills
        ]

        if required_skills:
            score = round(
                len(matched) / len(required_skills) * 100
            )
        else:
            score = 0
        recommendations.append({
            "title": job["title"],
            "company": job["company"],
            "description": job["description"],
            "score": score,
            "matched_skills": matched,
            "missing_skills": missing
        })

    recommendations.sort(
        key=lambda job: job["score"],
        reverse=True
    )

    return recommendations[:3]


# ---------- HOME ----------

@app.route("/", methods=["GET", "POST"])
def home():

    extracted_text = ""
    analysis = None
    error = None
    job_description = ""
    recommendations = []

    if request.method == "POST":

        file = request.files.get("resume")
        job_description = request.form.get(
            "job_description", ""
        )

        if not file or file.filename == "":
            error = "Please upload a resume."

        elif Path(file.filename).suffix.lower() not in [".pdf", ".txt"]:
            error = "Please upload a PDF or TXT file."

        elif not job_description.strip():
            error = "Please paste a job description."

        else:

            safe_name = secure_filename(file.filename)

            if not safe_name:
                error = "Invalid filename."

            else:

                file_path = UPLOAD_FOLDER / safe_name
                file.save(str(file_path))

                try:

                    extracted_text = extract_text_from_file(
                        str(file_path)
                    )

                    if not extracted_text.strip():
                        error = "No readable text found in resume."

                    else:

                        analysis = analyze_resume(
                            extracted_text,
                            job_description
                        )

                        recommendations = recommend_jobs(
                            analysis["resume_skills"]
                        )
                except Exception as exc:

                    error = f"Analysis failed: {exc}"

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        analysis=analysis,
        error=error,
        job_description=job_description,
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(debug=True)