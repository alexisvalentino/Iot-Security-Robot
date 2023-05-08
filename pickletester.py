import cv2
import pickle
import numpy as np
import face_recognition

# Load the trained model from the pickle file
model_path = "C:/Users/ACER/Desktop/Real time threat detection/models/model.pkl"
labels_path = "C:/Users/ACER/Desktop/Real time threat detection/models/labels.pkl"
print("Loading model and labels...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(labels_path, 'rb') as f:
    labels = pickle.load(f)

# Load the known faces model from the pickle file
known_faces_model_path = "C:/Users/ACER/Desktop/Real time threat detection/models/known_faces.pkl"
print("Loading known faces model...")
with open(known_faces_model_path, 'rb') as f:
    known_faces = pickle.load(f)

# Start the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not capture image.")
        break

    # Preprocess the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_frame = cv2.resize(gray, (100, 100))
    small_frame = np.array([small_frame])
    small_frame = small_frame.astype('float32') / 255.0
    small_frame = np.expand_dims(small_frame, axis=-1)

    # Use the model to predict the label of the new image
    predicted_label = model.predict(small_frame)
    predicted_person_name = labels[np.argmax(predicted_label)]

    # Check if the predicted person is a criminal or not
    if predicted_person_name in known_faces:
        # The person is in the known faces model
        known_face_embeddings = known_faces[predicted_person_name]
        face_locations = face_recognition.face_locations(frame, model="cnn")

        if len(face_locations) > 0:
            # Compute the face embeddings for the new image
            face_embeddings = face_recognition.face_encodings(frame, face_locations)

            # Compare the face embeddings to the known faces
            matches = face_recognition.compare_faces(known_face_embeddings, face_embeddings)

            if True in matches:
                # Generate an alert and send a notification
                print("Alert! Criminal detected.")

        # Draw a rectangle around the face
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Threat Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close the window
cap.release()
cv2.destroyAllWindows()
