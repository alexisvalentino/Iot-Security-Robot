# IoT Security Robot 🤖

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)]()

An intelligent, IoT-powered security robot designed for real-time threat detection, comprehensive surveillance, and remote monitoring. This project integrates computer vision, acoustic analysis, and mobile/email alerting systems to provide a proactive security solution.

## 🚀 Key Features

- **⚔️ Weapon Detection**: Uses OpenCV cascade classifiers to identify guns and knives in real-time.
- **👤 Face Recognition**: Biometric identification using the `face_recognition` library to distinguish between authorized users and unknown intruders.
- **🔊 Acoustic Monitoring**: Detects sudden loud noises or impacts by monitoring RMS audio levels.
- **📍 GPS Tracking**: Integrated GPS module support for geolocation data, crucial for mobile robot tracking.
- **📧 Multi-Channel Alerts**: 
  - **Email**: Sends detailed reports including screenshots of the threat, audio recordings, and Google Maps links.
  - **Telegram**: Instant notifications via a dedicated bot.
  - **Local Alarm**: Audible siren triggered via on-board speakers.
- **🌙 Night Vision**: Optimized for low-light surveillance conditions.

## 🛠️ Technical Stack

- **Core**: Python 3.6+
- **Vision**: OpenCV (`cv2`), `face_recognition`, `imutils`
- **Audio**: `sounddevice`, `soundfile`, `pygame`
- **Location**: `ublox`, `geocoder`, `opencage`
- **Communication**: `smtplib`, `python-telegram-bot`

## ⚙️ Setup & Installation

### 1. Prerequisites
Ensure you have Python installed. Then, install the necessary dependencies:

```bash
pip install opencage ublox pygame sounddevice soundfile face_recognition imutils geocoder python-telegram-bot opencv-python numpy
```

### 2. Configuration
Open `version2.py` and configure your credentials:

- **Email Settings**:
  ```python
  FROM_EMAIL = 'your-email@gmail.com'
  FROM_PASSWORD = 'your-app-password'
  TO_EMAIL = 'recipient-email@gmail.com'
  ```
- **Telegram Settings**:
  ```python
  TELEGRAM_BOT_TOKEN = 'your_bot_token'
  RECIPIENT_CHAT_ID = 'your_chat_id'
  ```
- **API Keys**:
  - Add your OpenCage API key for reverse geocoding.

### 3. File Requirements
Ensure the following files are in the root directory:
- `haarcascade_frontalface_default.xml` (Face detection)
- `cascade.xml` (Gun detection)
- `knife.xml` (Knife detection)
- `alarm.wav` (Siren sound)

## 🎮 Usage

Run the main application:
```bash
python version2.py
```

### Controls:
- **`q`**: Exit the security system.
- **`s`**: Manually silence the alarm.

## 📂 Project Structure

- `version2.py`: The main multi-threaded application (latest).
- `pickletrainer.py`: Utility to train faces for recognition.
- `pickletester.py`: Test face recognition accuracy.
- `haarcascade_*.xml`: Pre-trained models for object detection.

---
> [!IMPORTANT]
> This repository contains the **Python threat detection codebase only**. The mobile application codebase and the robot's secondary operating system codes are not included in this repository.

## 🎓 Future Development & Student Guide

For students or developers looking to iterate on this project, here are several recommended improvements to modernize the stack and enhance performance:

### 1. Modern Computer Vision
Currently, the system uses Haar Cascade classifiers. While lightweight, they are prone to false positives.
- **Switch to YOLO (v8+)**: Significantly higher accuracy and speed for weapon detection.
- **MediaPipe**: Consider for faster, more robust face and landmark detection.
- **Roboflow**: Use for managing datasets and training custom models if you add new threat types.

### 2. Improved Architecture
- **Asynchronous Programming**: Replace the current `threading` model with Python's `asyncio` for better resource management and less complexity.
- **Secrets Management**: Move API keys and email credentials out of the source code and into a `.env` file using `python-dotenv`.
- **NVR Integration**: Consider using RTSP streams for more professional-grade camera integration.

### 3. User Interface
- **Web Dashboard**: Instead of `cv2.imshow`, build a real-time web interface using **Streamlit**, **Flask**, or **FastAPI**. This allows for remote monitoring from any browser without needing the Python environment locally.

### 4. Cloud Integration
- **Google Cloud Vision**: For advanced analysis if local hardware is limited.
- **Firebase**: Use for real-time data syncing between the robot and the mobile app.


