from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
import datetime
import pygame
import sounddevice as sd
import soundfile as sf
import time
import smtplib
import threading
import cv2
import numpy as np
import imutils

FROM_EMAIL = 'f6866666@gmail.com'
FROM_PASSWORD = 'callktgjyogxqbwl'
TO_EMAIL = 'alexis01valentino@gmail.com'

gun_cascade = cv2.CascadeClassifier('cascade.xml')
camera = cv2.VideoCapture(0)

firstFrame = None
gun_exist = False
alarm_active = False

pygame.init()
alarm_sound = pygame.mixer.Sound('alarm.wav')

gun_detection_counter = 0
loud_sound_counter = 0

samplerate = sd.query_devices('Microphone (Realtek High Definition Audio), Windows DirectSound')['default_samplerate']
duration = 3  # seconds
device = sd.default.device

frame = None

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
                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        time.sleep(0.1)

# Sound detection
def calculate_rms(samples):
    return np.sqrt(np.mean(samples**2))

def loud_sound_detection_thread():
    global loud_sound_counter
    global alarm_sound
    global alarm_active
    rms_threshold = 0.1  # Set the threshold for RMS
    device = sd.default.device  # Get the default input/output device

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


# email
def email_sending_thread():
    global gun_exist
    global loud_sound_detected
    global frame
    global gun_detection_counter

    while True:
        if gun_detection_counter == 1 or loud_sound_counter == 1:
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
            text = MIMEText("Threat detected at " + datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"))
            msg.attach(text)
            with open(screenshot, 'rb') as f:
                img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename="screenshot.jpg")
            msg.attach(img)
            with open(filename, 'rb') as f:
                audio = MIMEAudio(f.read())
            audio.add_header('Content-Disposition', 'attachment', filename="recording.wav")
            msg.attach(audio)

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(FROM_EMAIL, FROM_PASSWORD)
            server.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())
            server.quit()

            # Reset counters
            gun_detection_counter = 0
            loud_sound_detected = False

        time.sleep(0.1)

# Create threads and start them
t1 = threading.Thread(target=gun_detection_thread)
t2 = threading.Thread(target=loud_sound_detection_thread)
t3 = threading.Thread(target=email_sending_thread)

t1.start()
t2.start()
t3.start()

while True:
    ret, frame = camera.read()

    # Copy the frame to avoid overwriting
    current_frame = frame.copy()

    current_frame = imutils.resize(current_frame, width=500)

    if firstFrame is None:
        firstFrame = current_frame.copy()
        continue

    cv2.putText(current_frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"), (10, current_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow("Security Feed", current_frame)
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

camera.release()
cv2.destroyAllWindows()
pygame.quit()

