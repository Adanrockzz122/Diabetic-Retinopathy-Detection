# Diabetic-Retinopathy-Detection
# 🩺 Diabetic Retinopathy Detection with Grad-CAM & Flask

A deep learning-based web application for detecting Diabetic Retinopathy (DR) from retinal fundus images using a custom ResNet-18 architecture and Grad-CAM for AI explainability. Built with PyTorch, Flask, and a beautifully styled, interactive frontend.

---

## 📁 Table of Contents

- [About the Project](#about-the-project)  
- [Dataset](#dataset)  
- [Technologies Used](#technologies-used)  
- [Model Architecture](#model-architecture)  
- [Explainability with Grad-CAM](#explainability-with-grad-cam)  
- [Flask Web App Features](#flask-web-app-features)  
- [Screenshots](#screenshots)  
- [Getting Started](#getting-started)  
- [Project Structure](#project-structure)  
- [Future Enhancements](#future-enhancements)  
- [License](#license)

---

## 📌 About the Project

Diabetic Retinopathy (DR) is a severe complication of diabetes that can lead to blindness. Early detection is key. This project provides a complete pipeline from image upload to prediction with confidence score and AI explanation using Grad-CAM—wrapped in a user-friendly web interface.

---

## 📊 Dataset

We use the Kaggle dataset:  
🔗 [Diagnosis of Diabetic Retinopathy Dataset](https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy)

- **Classes**: DR, No_DR  
- **Input Format**: Retinal fundus images  
- **Used for**: Binary classification

---

## 🧰 Technologies Used

| Technology     | Purpose                          |
|----------------|----------------------------------|
| PyTorch        | Model training and prediction    |
| torchvision    | Pretrained models and transforms |
| OpenCV         | Grad-CAM visualization           |
| Flask          | Backend web framework            |
| HTML/CSS/JS    | Frontend and interactivity       |
| Bootstrap      | UI styling and responsiveness    |
| Grad-CAM       | AI explainability                |

---

## 🧠 Model Architecture

We fine-tuned a **ResNet-18** model with a custom classifier head for binary classification.

```python
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)
```

- Trained with image augmentation  
- Model saved as `.pth` and integrated in Flask  
- Optimized for CPU inference

---

## 🔍 Explainability with Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is integrated to highlight **important regions** in the input image that contributed to the model's prediction.

- ✅ Heatmap overlays  
- ✅ Visual feedback for users  
- ✅ Built with forward and backward hooks on ResNet layer

---

## 🌐 Flask Web App Features

- 🖼️ Drag & Drop image upload  
- ⏳ Cool animated spinner during prediction  
- 🔥 Grad-CAM visual overlay  
- 📊 Animated confidence bar  
- 💬 Error handling with flash messages  
- 📱 Responsive and styled UI  
- 🎨 Background image and smooth transitions

---

## 📸 Screenshots

### 🔵 Before Prediction

![Before Prediction](static/sample-before.png)

### 🟢 After Prediction

![After Prediction](static/sample-after.png)

> _Update these paths once you save your screenshots in the `static/` folder._

---

## ✨ Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/diabetic-retinopathy-detector.git
cd diabetic-retinopathy-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Ensure Python 3.8+ and `torch`, `torchvision`, `Flask`, `Pillow`, `opencv-python` are installed.

### 3. Start the Server

```bash
python app.py
```

Then open: `http://127.0.0.1:5000` in your browser 🚀

---

## 📆 Project Structure

```
├── app.py
├── model/
│   ├── model.py
│   └── retinopathy_model.pth
├── static/
│   ├── background.jpg
│   ├── spinner.gif
│   ├── gradcam.jpg
│   ├── sample-before.png
│   └── sample-after.png
├── templates/
│   └── index.html
├── uploads/
└── requirements.txt
```

---

## 🌟 Future Enhancements

- [ ] Add multi-class classification (e.g., DR severity levels)  
- [ ] Deploy with Docker / on cloud platforms like Heroku or Render  
- [ ] Add patient report generation (PDF export)  
- [ ] Allow multiple image uploads  
- [ ] Track image upload history  
- [ ] Add retina segmentation or preprocessing options

---

## 📄 License

This project is licensed under the MIT License.  
Feel free to fork and build on it!

---

