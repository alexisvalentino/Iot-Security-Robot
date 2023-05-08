import os
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import sys
from scipy.spatial.distance import cosine
import tensorflow.compat.v1 as tf_v1
import pickle

def preprocess_face(face, required_size=(160, 160)):
    image = cv2.resize(face, required_size)
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0)
    return image

def load_facenet_model(model_path):
    with tf_v1.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf_v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    with tf_v1.Graph().as_default() as graph:
        tf_v1.import_graph_def(graph_def, name='')
    
    return graph

def get_face_embeddings(faces, image, graph):
    embeddings = []

    with tf_v1.Session(graph=graph) as sess:
        for face in faces:
            x1, y1, width, height = face['box']
            x2, y2 = x1 + width, y1 + height
            face_image = image[y1:y2, x1:x2]
            face_image = preprocess_face(face_image)
            embedding = sess.run(graph.get_tensor_by_name('embeddings:0'), feed_dict={
                graph.get_tensor_by_name('input:0'): face_image,
                graph.get_tensor_by_name('phase_train:0'): False
            })
            embeddings.append(embedding)

    return embeddings

def compare_faces(known_embeddings, unknown_embedding, threshold=0.5):
    matches = []
    for idx, known_embedding in enumerate(known_embeddings):
        distance = cosine(known_embedding, unknown_embedding)
        if distance <= threshold:
            matches.append(idx)
    return matches

# Load the MTCNN model for face detection
detector = MTCNN()

# Load the FaceNet model for face recognition
model_path = r'C:/Users/ACER/Desktop/Real time threat detection/models/20180402-114759.pb'
facenet_model = load_facenet_model(model_path)

# Load the known faces and their embeddings from a pickle file
known_faces_pickle = 'C:/Users/ACER/Desktop/Real time threat detection/known_faces/known_faces.pkl'
with open(known_faces_pickle, 'rb') as f:
    known_faces_data = pickle.load(f)

known_face_embeddings = known_faces_data['embeddings']
known_face_names = known_faces_data['names']

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)

    face_embeddings = get_face_embeddings(faces, frame_rgb, facenet_model)

    for i, unknown_face_embedding in enumerate(face_embeddings):
        matches = compare_faces(known_face_embeddings, unknown_face_embedding)
        if len(matches) > 0:
            for match_idx in matches:
                print(f"Match found: {known_face_names[match_idx]}")
                x, y, w, h = faces[i]['box']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, known_face_names[match_idx], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("Unknown face detected")
            x, y, w, h = faces[i]['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame with the detected faces
    cv2.imshow("Face Recognition", frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
