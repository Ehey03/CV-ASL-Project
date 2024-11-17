# inference.py
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
# from Model_Training import load_model

# Load the trained model

def load_model():
    return tf.keras.models.load_model('asl_model.h5')

model = load_model()

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Label mapping (assuming labels are 'A', 'B', ..., 'Z')
int_to_label = {i: chr(65 + i) for i in range(29)}  # Mapping integer to letters

# Start the webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to get hand landmarks
    results = hands.process(image_rgb)

    # Extract landmarks and predict
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
                landmarks.append(landmark.z)
            # Prepare the landmarks for prediction
            landmarks = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks)
            predicted_class = np.argmax(prediction)

            # Get the predicted letter
            predicted_letter = int_to_label[predicted_class]

            # Draw the hand landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the predicted letter on the frame
            cv2.putText(frame, predicted_letter, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)

    # Display the image
    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

