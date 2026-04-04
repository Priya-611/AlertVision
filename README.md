# 🚗 DriveGuard AI – Driver Drowsiness Detection

A real-time system that detects driver drowsiness using **OpenCV and Deep Learning (MobileNet)** and triggers an alert if the driver appears sleepy.

---

## 🚀 Features

* Real-time webcam detection
* Face and eye region extraction
* Deep learning classification (Awake / Sleepy)
* Time-based drowsiness detection
* Alarm alert system

---

## 🧠 How It Works

1. Capture video using webcam
2. Detect face using Haar Cascade
3. Extract eye region
4. Resize to 224×224 and normalize
5. Predict using trained model
6. If sleepy for more than 5 seconds → trigger alert

---

## 🏗️ Model

* Base: MobileNet (pretrained)
* Type: Binary classification
* Output:

  * 0 → Awake
  * 1 → Sleepy

---

## ▶️ Usage

### Train model

```bash
python src/train_model.py
```

### Run detection

```bash
python src/detect_drowsiness.py
```

Press **q** to exit.

---

## ⚠️ Notes

* Works best in good lighting
* Designed for Windows (uses winsound)

---

## 📌 Output

* Green → Awake
* Orange → Sleepy
* Red → Alert
