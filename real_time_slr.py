import cv2
import time
from collections import deque
import mediapipe as mp
from slr import extract_features, model


# Label map for predictions
label_map = [chr(ord('A') + i) for i in range(26)] + [' ', '<']

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam
print("Initializing webcam...")
cap = cv2.VideoCapture(0)

# Try alternate backend if first attempt fails
if not cap.isOpened():
    print("Trying alternate webcam backend...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set webcam properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

# Variables for stable prediction
text_output = ''
predictions = deque(maxlen=10)
last_added_pred = None
last_prediction_time = time.time()
last_action_time = time.time()  # Track last action time
cooldown_period = 1.0  # Seconds between registering predictions

print("Ready! Press 'q' to quit, 'c' to clear text")

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Flip horizontally for more natural interaction
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(rgb_frame)
    
    # Display text output
    cv2.rectangle(frame, (10, 10), (630, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Text: {text_output}", (15, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    current_prediction = None
    
    # Process hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract features
            landmarks = extract_features(hand_landmarks)
            
            # Make prediction
            prediction = model.predict([landmarks])[0]
            current_prediction = label_map[prediction]
            
            # Get prediction probabilities
            probs = model.predict_proba([landmarks])[0]
            confidence = probs[prediction]
            
            # Add to prediction history
            predictions.append(current_prediction)
            
            # Display prediction
            cv2.putText(frame, f"Predicted: {current_prediction} ({confidence:.2f})",
                       (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show cooldown timer
            elapsed = time.time() - last_prediction_time
            if elapsed < cooldown_period:
                # Draw cooldown progress bar
                progress = elapsed / cooldown_period
                bar_width = 200
                cv2.rectangle(frame, (15, 100), (15 + int(bar_width * progress), 115), 
                             (0, 255, 255), -1)
                cv2.rectangle(frame, (15, 100), (15 + bar_width, 115), 
                             (255, 255, 255), 1)
    
    # Process stable predictions
    if predictions and time.time() - last_prediction_time > cooldown_period:
        # Get most common prediction
        pred_counts = {}
        for p in predictions:
            pred_counts[p] = pred_counts.get(p, 0) + 1
        
        if pred_counts:
            most_common = max(pred_counts, key=pred_counts.get)
            most_common_count = pred_counts[most_common]
            
            # Only update if prediction is stable (appears in >60% of history)
            if most_common_count / len(predictions) > 0.6:
                # Update text based on prediction
                if most_common == '<':  # Backspace
                    if text_output:
                        text_output = text_output[:-1]
                else:
                    # Only add if different from last added (prevents duplicates)
                    if most_common != last_added_pred:
                        text_output += most_common
                        last_added_pred = most_common
                
                # Reset timer and prediction history
                last_prediction_time = time.time()
                predictions.clear()
                
                # Update last action time after a successful prediction
                last_action_time = time.time()
                
                # Visual feedback
                cv2.putText(frame, "ACCEPTED", (200, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Check for inactivity to add space
    if time.time() - last_action_time > 5.0:
        if text_output and text_output[-1] != ' ':
            text_output += ' '  # Add a space only if the last character is not a space
        last_action_time = time.time()  # Reset last action time

    # Show instructions
    cv2.putText(frame, "Press 'q' to quit, 'c' to clear", 
               (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Display frame
    cv2.imshow('ASL Recognition', frame)
    
    # Check for key presses
    key = cv2.waitKey(3) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        text_output = ''
        predictions.clear()

# Clean up
print("Closing program...")
hands.close()
cap.release()
cv2.destroyAllWindows()
print("Final text output:", text_output)