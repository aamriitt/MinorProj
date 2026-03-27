# 🧓 CareWatch – Elderly Monitoring System

## 🚀 Overview

CareWatch is a real-time elderly monitoring system designed to ensure the safety and well-being of senior citizens. It uses computer vision and alert mechanisms to detect inactivity and notify caregivers instantly.

---

## ✨ Features

* 🎥 **Real-time monitoring using webcam**
* 🧠 **Motion detection system**
* ⚠️ **Inactivity alerts (no movement detection)**
* 🔊 **Text-to-Speech (TTS) alerts**
* 📧 **Email notification system**
* 🌐 **Live dashboard using Flask**
* 🔌 **Real-time updates using Flask-SocketIO**

---

## 🛠️ Tech Stack

* Python 🐍
* Flask
* Flask-SocketIO
* OpenCV
* Yagmail (Email alerts)
* HTML, CSS (Dashboard)

---

## 📂 Project Structure

```
CareWatch/
│── app.py
│── templates/
│     └── dashboard.html
│── static/
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/aamriitt/MinorProj.git
cd MinorProj
```

### 2. Create virtual environment

```
python -m venv venv
```

### 3. Activate environment

```
venv\Scripts\Activate
```

### 4. Install dependencies

```
python -m pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## ⚠️ Notes

* Ensure webcam access is enabled
* Close other apps using the camera
* For email alerts, configure Yagmail credentials

---

## 📸 Future Improvements

* Mobile app integration 📱
* Fall detection using AI 🤖
* Emergency SOS button 🚨
* Cloud-based monitoring ☁️

---

## 👨‍💻 Author

**Amrit Dhal**

---

## 📜 License

This project is licensed under the MIT License.
