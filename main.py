import os
import cv2
import dlib
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# Directory containing test videos
VIDEOS_DIR = 'videos'

# Attempt to load your Keras model
model_path = 'model.keras'
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    model_accuracy = 0.95  # Replace with the actual model accuracy from training
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_accuracy = 0.0

# Serve video files from the 'videos' directory
@app.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory(VIDEOS_DIR, filename)

# Endpoint to list videos in the VIDEOS_DIR
@app.route('/list_videos', methods=['GET'])
def list_videos():
    videos = [f for f in os.listdir(VIDEOS_DIR) if f.endswith('.mp4')]
    return jsonify(videos=videos)

# Preprocessing and detection function
def process_video(video_path):
    if model is None:
        return {'label': 'Error loading model', 'accuracy': 0}

    detector = dlib.get_frontal_face_detector()
    
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5)  # Get frame rate
    result = "No faces detected"
    
    frame_count = 0
    detected_faces = 0
    predictions = []

    while cap.isOpened():
        frameId = cap.get(1)  # Get current frame number
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % ((int(frameRate) + 1) * 1) == 0:
            frame_count += 1
            face_rects, scores, idx = detector.run(frame, 0)
            for i, d in enumerate(face_rects):
                detected_faces += 1
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                crop_img = frame[y1:y2, x1:x2]
                resized_img = cv2.resize(crop_img, (128, 128))
                
                # Preprocess the image for the model
                resized_img = resized_img.astype('float32') / 255.0
                resized_img = np.expand_dims(resized_img, axis=0)
                
                # Predict using the Keras model
                prediction = model.predict(resized_img)
                label = 'FAKE' if prediction[0][0] > 0.5 else 'REAL'
                predictions.append(label)
                print(f"Processed frame {frameId}, Detected faces: {detected_faces}, Prediction: {label}")

    cap.release()
    print(f"Total frames processed: {frame_count}, Total faces detected: {detected_faces}")

    if predictions:
        # Determine final label based on majority voting
        final_label = max(set(predictions), key=predictions.count)
        return {'label': final_label, 'accuracy': model_accuracy * 100}

    return {'label': result, 'accuracy': 0}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_selected', methods=['POST'])
def process_selected():
    video_name = request.form['video']
    video_path = os.path.join(VIDEOS_DIR, video_name)
    result = process_video(video_path)
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
