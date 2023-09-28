# Iot Security Robot with Real time Threat Detection and Alerts, Night Vision and Mobile Integration.

<p>Traditional security systems, such as CCTV cameras, often have limitations in coverage, real-time threat detection, and responsiveness. These limitations can create a false sense of security, making it challenging to prevent crimes and protect people and property effectively.</p>
<p>To address these limitations, researchers have proposed a new type of security system: an IoT-powered smart security robot with an Android app, night vision, and enhanced threat detection capabilities. This prototype robot features an array of cameras for multifaceted surveillance, seamless mobile integration for remote monitoring and control, sophisticated microprocessors driving the robot's operating system, and employs state-of-the-art computer vision and machine learning algorithms for real-time threat identification, encompassing guns and knives, and loud noises. Additionally, the prototype robot can send real-time alerts to users through sound alarms and emails, complete with attached screenshots of detected threats (e.g., guns and knives), audio recordings of loud sounds, precise date and time stamps, and geolocation data indicating where the detection occurred.</p>

<h2>Features</h2>
<li>Gun and Knife Detection: The system uses a cascade classifier to detect the presence of guns or knives in the camera feed. If a gun or knife is detected, an alert is triggered.</li>
<li>Loud Sound Detection: The code captures audio samples and calculates the root mean square (RMS) value to determine if a loud sound is detected. An alert is triggered if a loud sound surpasses a predefined threshold.</li>
<li>Face Recognition: The system uses the face_recognition library to recognize known faces. If an unknown face is detected, an alert is triggered. (still in development)</li>
<li>Real time Alerts: When any threat is detected, a sound alarm is triggered and an email containing a screenshot, date and timestamps, audio recording, and geolocation information is sent. Additionally, a Telegram message (in development) is sent to a specified chat group.</li>
<li>Mobile app: For Remote Monitoring and Control</li>
<li>Night Vision Capabilities: For low-light conditions</li>

<h2>Prerequisites</h2>
<li>Python 3.6 or later</li>
<li>Required libraries: pygame, mapbox, folium, sounddevice, soundfile, threading, face_recognition, cv2, numpy, imutils, socket, geocoder, telegram</li>

<h2>Setup</h2>
<ol>
<li>Clone or download the repository containing the code.</li>
<li>Install the required Python libraries by running the following command: `pip install opencage ublox pygame sounddevice soundfile face_recognition imutils geocoder python-telegram-bot`</li>
<li>Replace the placeholders in the code with your actual email credentials:</li>
<li>Ensure you have the required XML cascade classifier files:</li>
<li>Run the code using a Python interpreter:</li>
<li>The security system will start monitoring the camera feed and detecting threats based on the implemented mechanisms.</li>
</ol>

<h2>Usage</h2>
<li>Press the 'q' key to exit the security system.</li>
<li>Press the 's' key to stop the alarm sound.</li>

<h2>Important Notes</h2>
<li>Carefully configure the detection thresholds and parameters in the code to match your environment and preferences.</li>
<li>Test the system thoroughly to ensure proper detection and alert functionality.</li>
<li>Be cautious about privacy and legal considerations when implementing face recognition and audio recording features.</li>
