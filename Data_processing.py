# data_processing.py
import cv2
import numpy as np
import os
import mediapipe as mp

def generate_landmarks(dataset_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    data = []
    labels = []
    
    for label in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, label)
        if os.path.isdir(class_dir):
            for image_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, image_file)
                image = cv2.imread(img_path)
                
                # Check if the image is loaded correctly
                if image is None:
                    print(f"Warning: Image at {img_path} could not be loaded.")
                    continue  # Skip to the next image
                
                # Convert the image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image to get hand landmarks
                results = hands.process(image_rgb)
                
                # Extract landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.append(landmark.x)
                            landmarks.append(landmark.y)
                            landmarks.append(landmark.z)
                        data.append(landmarks)
                        labels.append(label)  # Append the class label (e.g., 'A', 'B', etc.)
    
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    return data, labels
