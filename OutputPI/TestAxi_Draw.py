import os
import cv2
import random
import math
from ultralytics import YOLO
from pyaxidraw import axidraw

# Initialize AxiDraw
ad = axidraw.AxiDraw()
ad.options.speed_pendown = 50  # Set maximum pen-down speed to 50%
ad.interactive()  # Set AxiDraw to interactive mode
ad.connect()

# Define the absolute path to the videos directory
VIDEOS_DIR = '/home/pi/Downloads/PotholeRasp-main/OutputPI'

# Define the absolute path to the input video file
video_path = os.path.join(VIDEOS_DIR, 'TestPlot.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

# Check if the video file exists
if not os.path.isfile(video_path):
    raise FileNotFoundError(f"Video file '{video_path}' not found.")

cap = cv2.VideoCapture(video_path)

# Check if video capture is successful
if not cap.isOpened():
    raise IOError("Error: Cannot open video capture.")

ret, frame = cap.read()

# Check if frame is read successfully
if frame is None:
    raise IOError("Error: Cannot read frame from the video.")

H, W, _ = frame.shape

# Define the path to your custom model weights file
custom_model_path = '/home/pi/Downloads/PotholeRasp-main/OutputPI/best.pt'

# Check if the model weights file exists
if not os.path.exists(custom_model_path):
    raise FileNotFoundError(f"Model weights file '{custom_model_path}' not found.")

# Load the custom model
model = YOLO(custom_model_path)

# Define the confidence threshold
confidence_threshold = 0.6

# Define the class names we are interested in
pothole_class_name = "Pothole-aJPT"

def draw_circle(ad, center_x, center_y, radius):
    # Function to draw a circle with the plotter
    steps = 100  # Number of steps for drawing the circle
    angle_step = 2 * math.pi / steps
    ad.moveto(center_x + radius, center_y)  # Move to the starting point
    for step in range(steps + 1):
        angle = step * angle_step
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        ad.lineto(x, y)

while ret:
    results = model(frame)

    pothole_detected = False

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
                score = box.conf.item()  # Confidence score
                class_id = int(box.cls.item())  # Convert class_id tensor to int
                class_name = model.names[class_id]  # Retrieve the class name for the current detection

                if score > confidence_threshold:  # Apply confidence threshold
                    if class_name == pothole_class_name:
                        pothole_detected = True
                        # Draw bounding box around detected pothole
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green bounding box
                        text = f"{class_name}: {score:.2f}"
                        cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # Move the plotter to a specific location and draw a circle
                        center_x = 100  # Example center coordinates
                        center_y = 100
                        radius = 50  # Example radius
                        draw_circle(ad, center_x, center_y, radius)

    if not pothole_detected:
        # Draw random lines while the model is running through frames
        x_start = random.uniform(0, 210)  # Adjust based on plotter's drawing area
        y_start = random.uniform(0, 297)  # Adjust based on plotter's drawing area
        x_end = random.uniform(0, 210)
        y_end = random.uniform(0, 297)
        ad.moveto(x_start, y_start)
        ad.lineto(x_end, y_end)

    # Display the frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

# Disconnect from the Axidraw plotter
ad.disconnect()
