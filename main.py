import os

import fitz
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def home():
    extracted_text = ""

    if request.method == "POST":
        file = request.files.get("resume")

        if file and file.filename.lower().endswith(".pdf"):
            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(file_path)

            document = fitz.open(file_path)

            for page in document:
                extracted_text += page.get_text()

            document.close()

    return render_template(
        "index.html",
        extracted_text=extracted_text
    )


if __name__ == "__main__":
    app.run(debug=True)