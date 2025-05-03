import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from collections import deque

SOURCE_DIR = "C:\\Users\\twish\\OneDrive\\Desktop\\GestureConnect\\GC - Models\\ASL"
MODEL_PATH = "asl_model_new.pkl"


print("Loading dataset and extracting features...")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

data = []
labels = []
processed_count = 0
skipped_count = 0

def extract_features(hand_landmarks):
    """Extract features from hand landmarks"""
    # Get basic x,y coordinates
    x_list = [lm.x for lm in hand_landmarks.landmark]
    y_list = [lm.y for lm in hand_landmarks.landmark]
    z_list = [lm.z for lm in hand_landmarks.landmark]
    
    # Calculate boundaries
    x_min, x_max = min(x_list), max(x_list)
    y_min, y_max = min(y_list), max(y_list)
    z_min, z_max = min(z_list), max(z_list)
    
    # Calculate ranges and prevent division by zero
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    z_range = max(z_max - z_min, 1e-6)
    
    # Extract features - normalized x,y,z coordinates
    landmarks = []
    for lm in hand_landmarks.landmark:
        # Normalize coordinates relative to the hand size
        landmarks.append((lm.x - x_min) / x_range)
        landmarks.append((lm.y - y_min) / y_range)
        landmarks.append((lm.z - z_min) / z_range)
    
    # Add distance features
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]
    
    # Add distances between fingertips and wrist
    landmarks.append(distance(thumb_tip, index_tip))
    landmarks.append(distance(thumb_tip, wrist))
    landmarks.append(distance(index_tip, wrist))
    landmarks.append(distance(middle_tip, wrist))
    landmarks.append(distance(ring_tip, wrist))
    landmarks.append(distance(pinky_tip, wrist))
    
    return landmarks

def distance(p1, p2):
    """Calculate 3D distance between two points"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# Process each label folder
for label in range(28):  # 0-25 for A-Z, 26 for space, 27 for delete
    dir_path = os.path.join(SOURCE_DIR, str(label))
    if not os.path.exists(dir_path):
        print(f"Directory not found for label {label}, skipping.")
        continue
    
    label_name = chr(ord('A') + label) if label < 26 else "SPACE" if label == 26 else "DELETE"
    print(f"Processing label {label} ({label_name})...")
    
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        img = cv2.imread(file_path)
        
        if img is None:
            skipped_count += 1
            continue
            
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            skipped_count += 1
            continue
            
        # Extract features from the first detected hand
        landmarks = extract_features(results.multi_hand_landmarks[0])
        
        data.append(landmarks)
        labels.append(label)
        processed_count += 1
        
        # Print progress every 100 files
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} files so far...")

# Close MediaPipe
hands.close()

print(f"Dataset loading complete: {processed_count} images processed, {skipped_count} images skipped.")



if len(data) == 0:
    print("Error: No valid data extracted. Check your dataset path and image files.")
    exit()

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# --- MODEL TRAINING ---
print(f"Training model with {len(X)} samples...")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

