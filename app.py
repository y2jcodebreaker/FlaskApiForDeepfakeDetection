from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
from skimage.feature import graycomatrix, graycoprops
import dlib
from PIL import Image
import tempfile
import pandas as pd
import time
import joblib
import logging

# Initialize the Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the base directory relative to the current file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Set temporary directory for processing
temp_dir = tempfile.mkdtemp()

# Directory to save GLCM features
features_dir = os.path.join(temp_dir, 'glcm_features')
os.makedirs(features_dir, exist_ok=True)

# Directories to save frames and faces
frames_dir = os.path.join(temp_dir, 'frames')
faces_dir = os.path.join(temp_dir, 'faces')
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(faces_dir, exist_ok=True)

# Lazy loading for models and other resources
def get_detector():
    if not hasattr(get_detector, "detector"):
        get_detector.detector = dlib.get_frontal_face_detector()
    return get_detector.detector

def get_scaler():
    if not hasattr(get_scaler, "scaler"):
        scaler_path = os.path.join(base_dir, 'scaler.pkl')
        get_scaler.scaler = joblib.load(scaler_path)
    return get_scaler.scaler

def get_blending_model():
    if not hasattr(get_blending_model, "blending_model"):
        blending_model_path = os.path.join(base_dir, 'blending_model_knn_random_forest.pkl')
        get_blending_model.blending_model = joblib.load(blending_model_path)
    return get_blending_model.blending_model

def get_rf_model():
    if not hasattr(get_rf_model, "rf_model"):
        rf_model_path = os.path.join(base_dir, 'rf_classifier.pkl')
        get_rf_model.rf_model = joblib.load(rf_model_path)
    return get_rf_model.rf_model

def get_gb_model():
    if not hasattr(get_gb_model, "gb_model"):
        gb_model_path = os.path.join(base_dir, 'gb_classifier.pkl')
        get_gb_model.gb_model = joblib.load(gb_model_path)
    return get_gb_model.gb_model

def get_knn_model():
    if not hasattr(get_knn_model, "knn_model"):
        knn_model_path = os.path.join(base_dir, 'knn_classifier.pkl')
        get_knn_model.knn_model = joblib.load(knn_model_path)
    return get_knn_model.knn_model

# Helper function to save the uploaded video file
def save_uploaded_video(file):
    video_path = os.path.join(temp_dir, "uploaded_video.mp4")
    file.save(video_path)
    return video_path

# Helper function to extract frames from video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_interval = int(fps)  # Capture frames at 1 FPS

    for i in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(frames_dir, f'frame_{i}.jpg')
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            # Release memory
            del frame
    cap.release()
    return frames

# Helper function to detect and save faces from frames
def detect_faces(frames):
    faces = []
    detector = get_detector()
    for i, frame_path in enumerate(frames):
        frame = cv2.imread(frame_path)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = detector(rgb_frame)
        for j, rect in enumerate(detected_faces):
            x, y, w, h = rect.left(), rect.top(), rect.right(), rect.bottom()
            x = max(0, x)
            y = max(0, y)
            w = min(rgb_frame.shape[1], w)
            h = min(rgb_frame.shape[0], h)
            face_image = rgb_frame[y:h, x:w]
            face_path = os.path.join(faces_dir, f'face_{i}_{j}.jpg')
            cv2.imwrite(face_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
            faces.append(face_path)
        # Release memory
        del frame, rgb_frame
    return faces

# Helper function to save features to a CSV file
def save_features_to_csv(features, filename, columns):
    filepath = os.path.join(features_dir, filename)
    df = pd.DataFrame(features, columns=columns)
    df.to_csv(filepath, index=False)

# Helper function to extract GLCM features
def extract_glcm_features(faces):
    distances = [1, 5, 10]
    columns = ['contrast', 'dissimilarity', 'energy', 'asm', 'correlation']
    features_dict = {distance: [] for distance in distances}
    
    for face_path in faces:
        face = cv2.imread(face_path)
        resized_face = cv2.resize(face, (128, 128))
        gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
        for distance in distances:
            glcm = graycomatrix(gray_face, [distance], [0, np.pi/4, np.pi/2, 3*(np.pi/4)], symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast').flatten().mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten().mean()
            energy = graycoprops(glcm, 'energy').flatten().mean()
            asm = graycoprops(glcm, 'ASM').flatten().mean()
            correlation = graycoprops(glcm, 'correlation').flatten().mean()
            features_dict[distance].append([contrast, dissimilarity, energy, asm, correlation])
        # Release memory
        del face, resized_face, gray_face, glcm
    
    for distance in distances:
        features_array = np.array(features_dict[distance]).mean(axis=0).reshape(1, -1)
        save_features_to_csv(features_array, f'glcm_features_distance_{distance}.csv', columns)

@app.route('/')
def home():
    return "Welcome to the app"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Measure the time at the start of the script
        script_start_time = time.time()

        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']

        # Save and process the uploaded video
        video_path = save_uploaded_video(video_file)

        # Measure the time before frame extraction
        frame_extraction_start_time = time.time()
        frames = extract_frames(video_path)
        frame_extraction_end_time = time.time()
        frame_extraction_time = frame_extraction_end_time - frame_extraction_start_time

        # Measure the time before face extraction
        face_extraction_start_time = time.time()
        faces = detect_faces(frames)
        face_extraction_end_time = time.time()
        face_extraction_time = face_extraction_end_time - face_extraction_start_time

        # Measure the time before GLCM feature extraction
        glcm_extraction_start_time = time.time()
        extract_glcm_features(faces)
        glcm_extraction_end_time = time.time()
        glcm_extraction_time = glcm_extraction_end_time - glcm_extraction_start_time

        # Read the CSV file for distance 1
        csv_file_path = os.path.join(features_dir, 'glcm_features_distance_1.csv')
        features_df = pd.read_csv(csv_file_path)
        features = features_df.values.astype(np.float32)  # Ensure the features are of type float32

        # Scale the features
        scaler = get_scaler()
        scaled_features = scaler.transform(features)

        # Get predictions from base models
        rf_model = get_rf_model()
        gb_model = get_gb_model()
        knn_model = get_knn_model()
        rf_pred = rf_model.predict_proba(scaled_features)[:, 1].reshape(-1, 1)
        gb_pred = gb_model.predict_proba(scaled_features)[:, 1].reshape(-1, 1)
        knn_pred = knn_model.predict_proba(scaled_features)[:, 1].reshape(-1, 1)

        # Combine the base model predictions
        meta_features = np.hstack((rf_pred, gb_pred, knn_pred))

        # Measure the time before prediction
        prediction_start_time = time.time()
        blending_model = get_blending_model()
        predicted_class = blending_model.predict(meta_features)[0]
        prediction_end_time = time.time()
        prediction_time = prediction_end_time - prediction_start_time

        # Measure the time at the end of the script
        script_end_time = time.time()
        total_time = script_end_time - script_start_time

        # Print the prediction and the elapsed time
        logging.info(f"Prediction: {predicted_class}")
        logging.info(f"Time taken for frame extraction: {frame_extraction_time:.4f} seconds")
        logging.info(f"Time taken for face extraction: {face_extraction_time:.4f} seconds")
        logging.info(f"Time taken for GLCM feature extraction: {glcm_extraction_time:.4f} seconds")
        logging.info(f"Time taken for prediction: {prediction_time:.4f} seconds")
        logging.info(f"Total time taken for script execution: {total_time:.4f} seconds")
        
        # Return the prediction result and time taken in JSON format
        return jsonify({
                        'prediction': int(predicted_class),
            'frame_extraction_time': frame_extraction_time,
            'face_extraction_time': face_extraction_time,
            'glcm_extraction_time': glcm_extraction_time,
            'prediction_time': prediction_time,
            'total_time': total_time
        })
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['GET'])
def process():
    return jsonify({
        'frames_dir': frames_dir,
        'faces_dir': faces_dir,
        'features_dir': features_dir
    })

def app_handler(request):
    with app.app_context():
        return app.full_dispatch_request()

# Do not run the server when deploying to Google Cloud Functions
if __name__ == '__main__':
    logging.info("Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    logging.info("Flask server started.")