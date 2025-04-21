# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from model.model import get_model, preprocess_image, generate_gradcam_with_confidence

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
model = get_model()
class_names = ["DR", "No_DR"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            flash("No image file selected.")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        try:
            input_tensor = preprocess_image(file_path)
            label_idx, confidence, gradcam_path = generate_gradcam_with_confidence(model, input_tensor, file_path)
            label = class_names[label_idx]
            confidence = round(confidence * 100, 2)
            return render_template("index.html", prediction=label, confidence=confidence, gradcam=gradcam_path)
        except Exception as e:
            flash(f"An error occurred during prediction: {e}")
            return redirect(request.url)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)