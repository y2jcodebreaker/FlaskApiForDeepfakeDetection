Flask API for Deepfake Detection

This API allows users to upload videos and detect whether the faces in the video are real or fake using a blend of machine learning models. The API uses GLCM (Gray Level Co-occurrence Matrix) features for texture analysis and a blending model for prediction.

Table of Contents

	1.	Installation
	2.	Usage
	3.	Endpoints
	4.	File Structure
	5.	Models
	6.	License

Installation

To set up the project locally, follow these steps:
	1.	Clone the repository:
		git clone https://github.com/your-repo/FlaskApiForDeepfakeDetection.git
	2.	Navigate to the project directory:
		cd FlaskApiForDeepfakeDetection
	3.  Install the required packages:
		pip install -r requirements.txt
	4.	Ensure the machine learning models (knn_classifier.pkl, rf_classifier.pkl, gb_classifier.pkl, blending_model_knn_random_forest.pkl, scaler.pkl) are present in the project directory.

Usage

To start the Flask server, run the following command:
	python app.py

The server will start on localhost:8080 by default. You can send requests to the available endpoints for video processing and prediction.

Endpoints

/

Method: GET
Description: Basic endpoint to confirm that the API is running.
Response: "Welcome to the app"

/predict

Method: POST
Description: This endpoint accepts a video file for deepfake prediction.
Request: Multipart form-data containing a video file.
Response: JSON object with the predicted class (0 for real, 1 for fake) and time taken for different processing steps (frame extraction, face detection, GLCM feature extraction, and prediction).

/process

Method: GET
Description: Returns the directories where frames, faces, and features are stored during processing.

File Structure

	•	app.py: The main Flask application.
	•	scaler.pkl: Pre-trained scaler for normalizing GLCM features.
	•	blending_model_knn_random_forest.pkl: Pre-trained blending model for combining KNN, RandomForest, and GradientBoosting predictions.
	•	rf_classifier.pkl, knn_classifier.pkl, gb_classifier.pkl: Pre-trained base classifiers.
	•	requirements.txt: List of required Python packages.

Models

The API uses three machine learning models: KNN, RandomForest, and GradientBoosting. The blending model combines the predictions from these base models for the final prediction.


