# import cv2
# import time
# from collections import deque
# import mediapipe as mp
# import os
# import sys

# # Import from your existing modules
# from slr import extract_features, model  
# from chatbot import rag_chain, get_fallback_response

# # Label map for predictions (as in your existing code)
# label_map = [chr(ord('A') + i) for i in range(26)] + [' ', '<']

# # Initialize MediaPipe
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(
#     static_image_mode=False, 
#     max_num_hands=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.5
# )

# def run_sign_language_chatbot():
#     """Main function to run the integrated sign language chatbot"""
#     print("\n===== Welcome to GestureConnect: Sign Language Chatbot 😊=====")
    
#     conversation_active = True
    
#     while conversation_active:
#         # Capture sign language input
#         user_message = capture_sign_language_input()
        
#         # Check if user wants to exit
#         if user_message.upper() == "EXIT":
#             print("\nEXIT signed. Ending conversation...")
#             conversation_active = False
#             continue
        
#         # Process with chatbot
#         if user_message:
#             print(f"\nYou signed: {user_message}")
            
#             try:
#                 # Process with chatbot
#                 answer = rag_chain.invoke(user_message)
                
#                 # Handle empty response with fallback
#                 if not answer or answer.strip() == "":
#                     answer = get_fallback_response()
                    
#                 print(f"\nBot: {answer}")
#                 print("\nReady for next sign input. Sign 'EXIT' to quit.")
                
#             except Exception as e:
#                 print(f"Error in chatbot processing: {e}")
#                 print("Bot: I'm having trouble understanding. Please try again.")
    
#     print("Thank you for using GestureConnect!")

# def capture_sign_language_input():
#     """Captures sign language input using the existing SLR implementation"""
#     # Initialize webcam
#     print("\nInitializing webcam for sign input...")
#     cap = cv2.VideoCapture(0)
    
#     # Try alternate backend if first attempt fails
#     if not cap.isOpened():
#         print("Trying alternate webcam backend...")
#         cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return ""
    
#     # Set webcam properties for better performance
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
#     # Variables for stable prediction
#     text_output = ''
#     predictions = deque(maxlen=10)
#     last_added_pred = None
#     last_prediction_time = time.time()
#     last_action_time = time.time()  # Track last action time
#     cooldown_period = 1.0  # Seconds between registering predictions
    
#     print("Ready! Sign your message. Press 'q' to send, 'c' to clear text")
    
#     while True:
#         # Capture frame from webcam
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture frame")
#             break
        
#         # Flip horizontally for more natural interaction
#         frame = cv2.flip(frame, 1)
        
#         # Convert to RGB for MediaPipe
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process with MediaPipe
#         results = hands.process(rgb_frame)
        
#         # Display text output
#         cv2.rectangle(frame, (10, 10), (630, 50), (0, 0, 0), -1)
#         cv2.putText(frame, f"Text: {text_output}", (15, 35), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
#         current_prediction = None
        
#         # Process hand landmarks if detected
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Draw hand landmarks on frame
#                 mp_drawing.draw_landmarks(
#                     frame,
#                     hand_landmarks,
#                     mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )
                
#                 # Extract features
#                 landmarks = extract_features(hand_landmarks)
                
#                 # Make prediction
#                 prediction = model.predict([landmarks])[0]
#                 current_prediction = label_map[prediction]
                
#                 # Get prediction probabilities
#                 probs = model.predict_proba([landmarks])[0]
#                 confidence = probs[prediction]
                
#                 # Add to prediction history
#                 predictions.append(current_prediction)
                
#                 # Display prediction
#                 cv2.putText(frame, f"Predicted: {current_prediction} ({confidence:.2f})",
#                            (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
#                 # Show cooldown timer
#                 elapsed = time.time() - last_prediction_time
#                 if elapsed < cooldown_period:
#                     # Draw cooldown progress bar
#                     progress = elapsed / cooldown_period
#                     bar_width = 200
#                     cv2.rectangle(frame, (15, 100), (15 + int(bar_width * progress), 115), 
#                                  (0, 255, 255), -1)
#                     cv2.rectangle(frame, (15, 100), (15 + bar_width, 115), 
#                                  (255, 255, 255), 1)
        
#         # Process stable predictions
#         if predictions and time.time() - last_prediction_time > cooldown_period:
#             # Get most common prediction
#             pred_counts = {}
#             for p in predictions:
#                 pred_counts[p] = pred_counts.get(p, 0) + 1
            
#             if pred_counts:
#                 most_common = max(pred_counts, key=pred_counts.get)
#                 most_common_count = pred_counts[most_common]
                
#                 # Only update if prediction is stable (appears in >60% of history)
#                 if most_common_count / len(predictions) > 0.6:
#                     # Update text based on prediction
#                     if most_common == '<':  # Backspace
#                         if text_output:
#                             text_output = text_output[:-1]
#                     else:
#                         # Only add if different from last added (prevents duplicates)
#                         if most_common != last_added_pred:
#                             text_output += most_common
#                             last_added_pred = most_common
                    
#                     # Reset timer and prediction history
#                     last_prediction_time = time.time()
#                     predictions.clear()
                    
#                     # Update last action time after a successful prediction
#                     last_action_time = time.time()
                    
#                     # Visual feedback
#                     cv2.putText(frame, "ACCEPTED", (200, 80), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
#         # Check for inactivity to add space
#         if time.time() - last_action_time > 5.0:
#             if text_output and text_output[-1] != ' ':
#                 text_output += ' '  # Add a space only if the last character is not a space
#             last_action_time = time.time()  # Reset last action time
    
#         # Show instructions
#         cv2.putText(frame, "Press 'q' to send to chatbot, 'c' to clear", 
#                    (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
#         # Display frame
#         cv2.imshow('Sign Language Chatbot', frame)
        
#         # Check for key presses
#         key = cv2.waitKey(3) & 0xFF
#         if key == ord('q'):
#             # Send message to chatbot
#             cap.release()
#             cv2.destroyAllWindows()
#             return text_output
#         elif key == ord('c'):
#             text_output = ''
#             predictions.clear()
    
#     # In case of premature exit
#     cap.release()
#     cv2.destroyAllWindows()
#     return text_output

# if __name__ == "__main__":
#     run_sign_language_chatbot()


import cv2
import time
from collections import deque
import mediapipe as mp
import os
import sys

# Import from your existing modules
from slr import extract_features, model  # Your existing SLR model
from chatbot import rag_chain, get_fallback_response  # Import chatbot components

# Label map for predictions (as in your existing code)
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

def main():
    """Main entry point that bypasses the initial text interface"""
    print("\n===== Welcome to GestureConnect: Sign Language Chatbot 😊 =====")
    
    # Print a welcome message from the chatbot
    print("\nBot: Hello! Welcome to GestureConnect! How can I assist you today?")
    
    # Start the SLR+chatbot conversation loop
    run_sign_language_chatbot()

def run_sign_language_chatbot():
    """Main function to run the integrated sign language chatbot"""
    conversation_active = True
    
    while conversation_active:
        # Capture sign language input
        user_message = capture_sign_language_input()
        
        # Check if user wants to exit
        if user_message.upper() == "EXIT":
            print("\nEXIT signed. Ending conversation...")
            conversation_active = False
            continue
        
        # Process with chatbot
        if user_message:
            print(f"\nYou signed: {user_message}")
            
            try:
                # Process with chatbot
                answer = rag_chain.invoke(user_message)
                
                # Handle empty response with fallback
                if not answer or answer.strip() == "":
                    answer = get_fallback_response()
                    
                print(f"\nBot: {answer}")
                print("\nReady for next sign input. Sign 'EXIT' to quit.")
                
            except Exception as e:
                print(f"Error in chatbot processing: {e}")
                print("Bot: I'm having trouble understanding. Please try again.")
    
    print("Thank you for using GestureConnect!")

def capture_sign_language_input():
    """Captures sign language input using the existing SLR implementation"""
    # Initialize webcam
    print("\nInitializing webcam for sign input...")
    cap = cv2.VideoCapture(0)
    
    # Try alternate backend if first attempt fails
    if not cap.isOpened():
        print("Trying alternate webcam backend...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return ""
    
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
    
    print("Ready! Sign your message. Press 'q' to send, 'c' to clear text")
    
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
        cv2.putText(frame, "Press 'q' to send to chatbot, 'c' to clear", 
                   (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Sign Language Chatbot', frame)
        
        # Check for key presses
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):
            # Send message to chatbot
            cap.release()
            cv2.destroyAllWindows()
            return text_output
        elif key == ord('c'):
            text_output = ''
            predictions.clear()
    
    # In case of premature exit
    cap.release()
    cv2.destroyAllWindows()
    return text_output

if __name__ == "__main__":
    main()  # Use the new main function as entry point