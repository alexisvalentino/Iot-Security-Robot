import pickle
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from opencage.geocoder import OpenCageGeocode
from twilio.rest import Client
import datetime
import pygame
import os
import smtplib
import sounddevice as sd
import soundfile as sf
import time
import threading
import face_recognition
import cv2
import numpy as np
import imutils
import socket
import geocoder

FROM_EMAIL = 'f6866666@gmail.com'
FROM_PASSWORD = 'gwlyhghawjphuwtv'
TO_EMAIL = 'alexis01valentino@gmail.com'

TWILIO_ACCOUNT_SID = 'your_twilio_account_sid'
TWILIO_AUTH_TOKEN = 'your_twilio_auth_token'
TWILIO_PHONE_NUMBER = 'your_twilio_phone_number'
RECIPIENT_PHONE_NUMBER = 'recipient_phone_number'

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gun_cascade = cv2.CascadeClassifier('cascade.xml')
camera = cv2.VideoCapture(0)

OPENCAGE_API_KEY = 'e4d21fe225704779a60d81989b3285b2'
firstFrame = None
gun_exist = False
alarm_active = False
unknown_face_detected = False
unknown_face_image = None

pygame.init()
alarm_sound = pygame.mixer.Sound('alarm.wav')
gun_detection_counter = 0
loud_sound_counter = 0
device_info = sd.query_devices(device=2)
samplerate = device_info['default_samplerate']
duration = 3  # seconds
device = sd.default.device
frame = None
known_face_encodings = []
known_face_names = []
unknown_face_detected = False
unknown_face_image = None
alarm_triggered = False

KNOWN_FACES_PICKLE = 'known_faces.pkl'
if os.path.exists(KNOWN_FACES_PICKLE):
    with open(KNOWN_FACES_PICKLE, 'rb') as f:
        known_faces = pickle.load(f)
    known_face_names = list(known_faces.keys())
    known_face_encodings = [np.array(encoding) for encoding in known_face_encodings for encoding in known_face_encodings]

# Gun detection
def gun_detection_thread():
    global gun_exist
    global frame
    global gun_detection_counter
    gun_alarm_time = None
    while True:
        if frame is not None:

            # Resize the frame and convert it to grayscale
            resized_frame = cv2.resize(frame, (500, 500))
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            # Detect guns in the frame using the cascade classifier
            guns = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

            # Set the gun_exist flag if guns are detected
            if len(guns) > 0:
                gun_exist = True
                gun_detection_counter += 1
                if gun_alarm_time is None or time.time() - gun_alarm_time >= 3:

                    # Alarm if gun detected for the first time or after 3 seconds
                    print("Gun detected!")
                    for i in range(5):

                        # Continuously alarm for at least 5 seconds
                        print("ALARM!")
                        time.sleep(1)
                    gun_alarm_time = time.time()

            else:
                gun_exist = False
                gun_detection_counter = 0
                gun_alarm_time = None

            # Draw a blue rectangle around each detected gun
            for (x, y, w, h) in guns:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

        time.sleep(0.1)

# Sound detection
def calculate_rms(samples):
    return np.sqrt(np.mean(np.nan_to_num(samples)**2))

def loud_sound_detection_thread():
    global loud_sound_detected
    global loud_sound_counter
    global alarm_sound
    global alarm_active
    rms_threshold = 0.4  # Set the threshold for RMS
    while True:
        samples = sd.rec(int(duration * samplerate), device=device, channels=1)
        sd.wait()
        rms_value = calculate_rms(samples)
        # Check if sound is loud enough and not the alarm sound
        if rms_value > rms_threshold and not alarm_active:
            print("loud sound detected")
            loud_sound_counter += 1
        else:
            loud_sound_counter = 0
        time.sleep(0.1)

#Face recognition
def face_recognition_thread():
    global frame
    global known_face_encodings
    global known_face_names
    global unknown_face_detected
    global unknown_face_image
    global alarm_triggered

    # Set the threshold for face recognition
    face_recognition_threshold = 0.1

    # Store the last recognized face distance and name
    last_face_distance = None   
    last_face_name = None

    # Set alarm_triggered flag to False initially
    alarm_triggered = False
    while True:
        if frame is not None:
            small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            if len(face_encodings) > 0:
                # Track faces using face_distance() function
                face_distances = []
                for face_encoding in face_encodings:
                    face_distances.append(face_recognition.face_distance(known_face_encodings, face_encoding))

                # Find the closest match for each face
                best_match_names = []
                for distances in face_distances:
                    if len(distances) > 0:  # Check if the distances list is not empty
                        best_match_index = np.argmin(distances)
                        if distances[best_match_index] < face_recognition_threshold:
                            best_match_names.append(known_face_names[best_match_index])
                        else:
                            best_match_names.append("Unknown")
                    else:
                        best_match_names.append("Unknown")

                # Update the last recognized face distance and name
                if last_face_distance is None:
                    last_face_distance = face_distances[0]
                    last_face_name = best_match_names[0]

                else:
                    for i in range(len(face_distances)):
                        if face_distances[i][0] < last_face_distance[0]:
                            last_face_distance = face_distances[i]
                            last_face_name = best_match_names[i]

                # Draw a rectangle and label for each detected face
                for (top, right, bottom, left), name in zip(face_locations, best_match_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Draw a label with the name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                 # Set off the alarm if an unknown face is detected
                if "Unknown" in best_match_names:
                    if not alarm_triggered:  # Check if the alarm has not been triggered before
                        print('unknown face detected')
                        unknown_face_detected = True
                        unknown_face_image = frame.copy()

                        # Set off the alarm
                        alarm_thread()
                        alarm_triggered = True  # Set the alarm triggered flag to True
                else:
                    print('known face detected')
                    unknown_face_detected = False
                    alarm_triggered = False  # Reset the alarm triggered flag
            else:
                print('no face detected')
        time.sleep(0.1)

# Alarm
def alarm_thread():
    global alarm_active
    global unknown_face_detected
    sound_duration = 2  # Assuming the sound file has a 2-second duration
    loop_count = int(5 / sound_duration) - 1  # Calculate the loop count for a 5-second alarm

    while True:
        if unknown_face_detected and not alarm_active:
            alarm_sound.play(loop_count)
            alarm_active = True
            time.sleep(5)
            alarm_sound.stop()
            alarm_active = False
            unknown_face_detected = False
        time.sleep(0.1)

#sms
def send_sms(message):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )
    return message.sid

def sms_alert_thread():
    while True:
        # Check for conditions that require SMS alerts
        if gun_detection_counter == 1 or loud_sound_counter == 1 or unknown_face_detected:
            sms_message = "Threat detected - SMS alert."
            send_sms(sms_message)
        time.sleep(0.1)

#email
def email_sending_thread():
    global gun_detection_counter
    global loud_sound_counter
    global unknown_face_detected
    global unknown_face_image
    while True:
        if gun_detection_counter == 1 or loud_sound_counter == 1 or unknown_face_detected:
            # Record audio
            samples = sd.rec(int(duration * samplerate), device=device, channels=1)
            sd.wait()
            screenshot = "screenshot.jpg"
            cv2.imwrite(screenshot, frame)
            filename = "recording.wav"
            sf.write(filename, samples, int(samplerate))
            msg = MIMEMultipart()
            msg['From'] = FROM_EMAIL
            msg['To'] = TO_EMAIL
            msg['Subject'] = 'Threat Detected'
            # Get the device's public IP address using the socket library
            public_ip = socket.gethostbyname(socket.gethostname())

            try:
                # Use geocoder to get latitude and longitude of the IP address
                g = geocoder.ip(public_ip)
                latitude = g.latlng[0]
                longitude = g.latlng[1]
                # Construct a Google Maps link with the obtained coordinates
                google_maps_link = f"https://www.google.com/maps/place/{latitude},{longitude}"
                location_info = f"Location: {google_maps_link}"  # Use the Google Maps link as location
            except Exception as e:
                location_info = "Location: Unavailable"
            text = MIMEText(
                "Threat detected at " + datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p") + "\n" + location_info)
            msg.attach(text)

            # Attach the screenshot
            with open(screenshot, 'rb') as f:
                img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename="screenshot.jpg")
            msg.attach(img)

            # Attach the audio recording
            with open(filename, 'rb') as f:
                audio = MIMEAudio(f.read())
            audio.add_header('Content-Disposition', 'attachment', filename="recording.wav")
            msg.attach(audio)

            # Attach unknown face screenshot if applicable
            if unknown_face_detected:
                unknown_screenshot = "unknown_screenshot.jpg"
                cv2.imwrite(unknown_screenshot, unknown_face_image)
                with open(unknown_screenshot, 'rb') as f:
                    img_unknown = MIMEImage(f.read())
                img_unknown.add_header('Content-Disposition', 'attachment', filename="unknown_screenshot.jpg")
                msg.attach(img_unknown)

            try:
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(FROM_EMAIL, FROM_PASSWORD)
                server.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())
                server.quit()
                print("Email sent successfully")

            except Exception as e:
                print("Error sending email:", str(e))

        time.sleep(0.1)



def detect_faces_opencv(camera):
    global frame
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to read camera frame")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            print("No faces detected")
            continue

        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Create threads and start them
t1 = threading.Thread(target=gun_detection_thread)
t2 = threading.Thread(target=loud_sound_detection_thread)
t3 = threading.Thread(target=email_sending_thread)
t4 = threading.Thread(target=face_recognition_thread)
t5 = threading.Thread(target=alarm_thread)
t6 = threading.Thread(target=detect_faces_opencv, args=(camera,))
t7 = threading.Thread(target=sms_alert_thread)

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start() 

while True:
    ret, frame = camera.read()
    if not ret:  # Check if the camera frame is valid
        print("Error: Failed to read camera frame")
        break
    frame = imutils.resize(frame, width=500)
    if firstFrame is None:
        firstFrame = frame.copy()
        continue
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imshow("Security Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if gun_detection_counter > 0 or loud_sound_counter > 0:
        if not alarm_active:
            alarm_sound.play(-1)
            alarm_active = True

    elif alarm_active:
        alarm_sound.stop()
        alarm_active = False

    if key == ord('s'):
        if alarm_active:
            alarm_sound.stop()
            alarm_active = False

# Wait for threads to finish before closing
t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()

camera.release()
cv2.destroyAllWindows()
pygame.quit()