import numpy as np
import cv2
import imutils
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import datetime
import pygame

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

frames_since_detection = 0

while True:

    ret, frame = camera.read()

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gun = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    if len(gun) > 0:
        gun_exist = True
    else:
        gun_exist = False

    for (x, y, w, h) in gun:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

    if firstFrame is None:
        firstFrame = gray
        continue

    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow("Security Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if gun_exist:
        print("guns detected")
        frames_since_detection += 1
        if not alarm_active:
            alarm_sound.play(-1)
            alarm_active = True
        if frames_since_detection > 10:
            screenshot = "screenshot.jpg"
            cv2.imwrite(screenshot, frame)
            msg = MIMEMultipart()
            msg['From'] = FROM_EMAIL
            msg['To'] = TO_EMAIL
            msg['Subject'] = 'Gun Detected'
            text = MIMEText("Gun detected at " + datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"))
            msg.attach(text)
            with open(screenshot, 'rb') as f:
                img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename="screenshot.jpg")
            msg.attach(img)
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(FROM_EMAIL, FROM_PASSWORD)
            server.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())
            server.quit()
            frames_since_detection = 0
    else:
        print("guns NOT detected")
        frames_since_detection = 0
        if alarm_active:
            alarm_sound.stop()
            alarm_active = False

    if key == ord('s'):
        if alarm_active:
            alarm_sound.stop()
            alarm_active = False

camera.release()
cv2.destroyAllWindows()
pygame.quit()
