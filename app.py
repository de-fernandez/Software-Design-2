# app.py
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os

app = Flask(__name__)

MODEL_PATH = "model/tomato_yolo_best.pt"
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

CLASSES = ["Reject", "Ripe", "Unripe"]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
@app.route("/detect-page")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return redirect(url_for("home"))

    file = request.files["image"]

    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("home"))

    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    results = model(img_path)
    predicted_index = results[0].probs.top1
    result_label = CLASSES[predicted_index] if predicted_index < len(CLASSES) else "Unknown"

    if result_label == "Ripe":
        result_label = "Unripe"
    elif result_label == "Unripe":
        result_label = "Ripe"

    return render_template(
        "results.html",
        result=result_label
    )

if __name__ == "__main__":
    app.run(debug=False)