
import os
import re
import fitz
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SKILLS = [
    "python", "java", "c++", "sql", "excel", "power bi",
    "tableau", "machine learning", "deep learning",
    "data analysis", "data analytics", "statistics",
    "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "nlp", "html", "css", "javascript",
    "flask", "django", "git", "communication",
    "leadership", "problem solving", "project management",
    "aws", "azure", "mongodb", "mysql", "postgresql"
]


def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in SKILLS:
        pattern = r'(?<!\w)' + re.escape(skill) + r'(?!\w)'
        if re.search(pattern, text):
            found_skills.append(skill)

    return found_skills


@app.route("/", methods=["GET", "POST"])
def home():
    extracted_text = ""
    job_description = ""
    match_score = None
    matched_skills = []
    missing_skills = []
    message = ""

    if request.method == "POST":
        file = request.files.get("resume")
        job_description = request.form.get("job_description", "").strip()

        if not file or not file.filename:
            message = "Please upload a resume PDF."

        elif not file.filename.lower().endswith(".pdf"):
            message = "Only PDF files are supported."

        elif not job_description:
            message = "Please enter a job description."

        else:
            safe_filename = secure_filename(file.filename)

            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                safe_filename
            )

            file.save(file_path)

            try:
                document = fitz.open(file_path)

                for page in document:
                    extracted_text += page.get_text()

                document.close()

                resume_skills = extract_skills(extracted_text)
                jd_skills = extract_skills(job_description)

                matched_skills = [
                    skill for skill in jd_skills
                    if skill in resume_skills
                ]

                missing_skills = [
                    skill for skill in jd_skills
                    if skill not in resume_skills
                ]

                if jd_skills:
                    match_score = round(
                        (len(matched_skills) / len(jd_skills)) * 100,
                        2
                    )
                else:
                    message = (
                        "No recognizable skills were found "
                        "in the job description."
                    )

            except Exception as e:
                message = f"Error processing resume: {e}"

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        job_description=job_description,
        match_score=match_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
