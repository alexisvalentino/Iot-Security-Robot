import os
import pickle
import cv2
import numpy as np
import face_recognition
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical

# Step 1: Collect a dataset of facial images of individuals with a criminal background
dataset_path = "C:/Users/ACER/Desktop/Real time threat detection/known_faces"
images = []
labels = []

# Step 2: Label the dataset appropriately to indicate whether the individual has a criminal background or not
print("Processing dataset...")
for i, person_name in enumerate(os.listdir(dataset_path)):
    print(f"Processing {person_name}...")
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            print(f"Processing image {image_name}...")
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (100, 100))
            images.append(image)
            labels.append(i)

# Step 3: Preprocess the dataset to remove any noise or artifacts, and to normalize the images for lighting and pose
print("Preprocessing dataset...")
images = np.array(images)
images = images.astype('float32') / 255.0
images = np.expand_dims(images, axis=-1)

# Step 4: Split the dataset into training and validation sets, and use the training set to train the facial recognition model
print("Splitting dataset...")
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
num_classes = len(np.unique(labels))

# Step 5: Define the model architecture, either by using pre-existing models or by building your own deep learning model
print("Building model...")
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Step 6: Train the model and evaluate its accuracy on the validation set to fine-tune it
print("Training model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Step 7: Save the trained model as a pickle file
model_path = "C:/Users/ACER/Desktop/Real time threat detection/models/model.pkl"
labels_path = "C:/Users/ACER/Desktop/Real time threat detection/models/labels.pkl"
print("Saving model and labels...")
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
with open(labels_path, 'wb') as f:
    pickle.dump(labels, f)


# Step 8: Finally, use the trained model to detect individuals with a criminal background in real-time using the camera feed
# Load the model from the pickle file
print("Done.")
print("Loading model...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Initialize the known faces dictionary
known_faces = {}

# Iterate through the dataset folder
dataset_path = "C:/Users/ACER/Desktop/Real time threat detection/known_faces"
print("Processing dataset...")

for person_name in os.listdir(dataset_path):
    print(f"Processing {person_name}...")
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        known_faces[person_name] = []
        
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
                known_faces[person_name].append(face_embedding)

# Save the known faces model as a pickle file
known_faces_model_path = "C:/Users/ACER/Desktop/Real time threat detection/models/known_faces.pkl"
os.makedirs(os.path.dirname(known_faces_model_path), exist_ok=True)
print("Saving known faces model...")

with open(known_faces_model_path, 'wb') as f:
    pickle.dump(known_faces, f)

print("Done.")
