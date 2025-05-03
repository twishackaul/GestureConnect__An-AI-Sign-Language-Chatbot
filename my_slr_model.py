import numpy as np
import pickle

# Load the model when module is imported
try:
    with open('asl_model_new.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: SLR model file not found. Make sure 'asl_model_new.pkl' exists in the current directory.")
    exit()

def predict_sign(landmarks):
    """
    Predict the sign from landmarks
    
    Args:
        landmarks: List of landmark features extracted from hand detection
        
    Returns:
        int: Predicted class/label index (0-27)
    """
    landmarks_array = np.array([landmarks])  
    prediction = model.predict(landmarks_array)
    return prediction[0]