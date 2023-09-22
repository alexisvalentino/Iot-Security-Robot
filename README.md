# Codebase for the Threat Detection of my Research project Smart security robot
<p>It is an Iot Powered Smart Security Surveillance Robot with ML Based Real-time Threat Detection, Real time alerts with Event and Geolocation logging, Night Vision Capabilities and Android Application for Remote Monitoring and Control</p>

<p>This Python code implements a security system that combines various threat detection mechanisms including gun and knife detection, loud sound detection, and face recognition. The system captures live camera feed and processes it to detect potential threats. Upon detection, it triggers alerts via email that includes metadata as well as activates an alarm sound.</p>

<h2>Features</h2>
<li>Gun and knife Detection: The system uses a cascade classifier to detect the presence of guns or knives in the camera feed. If a gun or knife is detected, an alert is triggered.</li>
<li>Loud Sound Detection: The code captures audio samples and calculates the root mean square (RMS) value to determine if a loud sound is detected. An alert is triggered if a loud sound surpasses a predefined threshold.</li>
<li>Face Recognition: The system uses the face_recognition library to recognize known faces. If an unknown face is detected, an alert is triggered.</li>
<li>Email and Telegram Alerts: When any threat is detected, an email containing a screenshot, date and timestamps, audio recording, and location information is sent. Additionally, a Telegram message is sent to a specified chat group.</li>

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

<p>Feel free to customize and expand upon this README to provide more details about your project, its dependencies, setup instructions, and any other relevant information.</p>
