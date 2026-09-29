print("AI Resume Screening Project Initialized!")
from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return "AI Resume Screening System is running!"


if __name__ == "__main__":
    app.run(debug=True)
