Overview
Potholes are hazardous road defects that can cause vehicle damage and accidents. This Pothole Detection System leverages real-time computer vision and deep learning to automatically detect potholes from a live video feed. The system uses a YOLO-based object detection model to identify and localize potholes, providing instant feedback and statistics via a modern dashboard interface.

Features
Real-time pothole detection using a live camera feed.

YOLOv12 deep learning model for accurate and fast object detection.

Dashboard UI displaying live video, detection statistics, and system status.

Configurable confidence threshold for detections.

Automatic frame rate and detection count updates.
Requirements
Python 3.7+

Numpy

Tensorflow

Keras

OpenCV

Scikit-learn
Results
Achieved up to 94% detection accuracy on the test set using YOLOv7/YOLOv12.

Real-time detection at ~30 FPS (depending on hardware).

Displays number of detected potholes and confidence scores on the dashboard.
