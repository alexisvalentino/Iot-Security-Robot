import os
import pickle
import cv2
import numpy as np
import face_recognition

# Initialize the lists for embeddings and names
embeddings_list = []
names_list = []

dataset_path = "C:/Users/ACER/Desktop/Real time threat detection/known_faces"
print("Processing dataset...")

for person_name in os.listdir(dataset_path):
    print(f"Processing {person_name}...")
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        
        # Iterate through the images of each person
        for image_name in os.listdir(person_path):
            print(f"Processing image {image_name}...")
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Compute the face embedding for each image
            face_locations = face_recognition.face_locations(rgb_image, model="cnn")
            if face_locations:
                face_embedding = face_recognition.face_encodings(rgb_image, face_locations)[0]
                embeddings_list.append(face_embedding)
                names_list.append(person_name)

# Save the known faces model as a pickle file
known_faces_model_path = "C:/Users/ACER/Desktop/Real time threat detection/known_faces/known_faces.pkl"
os.makedirs(os.path.dirname(known_faces_model_path), exist_ok=True)
print("Saving known faces model...")

known_faces_data = {
    'embeddings': embeddings_list,
    'names': names_list
}

with open(known_faces_model_path, 'wb') as f:
    pickle.dump(known_faces_data, f)

print("Done.")
