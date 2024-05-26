import os
import cv2
import random
import math
import time
import threading
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
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

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
pothole_class_name = "pothole-ajpt".strip().lower()

# Function to move plotter to a specific location and wait for 2 seconds
def move_plotter_to_location(ad, x, y):
    ad.moveto(x, y)
    time.sleep(2)

# Function to draw random lines with the plotter
def draw_random_lines(ad, running):
    while running.is_set():
        x_start = random.uniform(0, 210)  # Adjust based on plotter's drawing area
        y_start = random.uniform(0, 297)  # Adjust based on plotter's drawing area
        x_end = random.uniform(0, 210)
        y_end = random.uniform(0, 297)
        ad.moveto(x_start, y_start)
        ad.lineto(x_end, y_end)

# Thread control
running = threading.Event()
running.set()
line_thread = threading.Thread(target=draw_random_lines, args=(ad, running))
line_thread.start()

while ret:
    results = model(frame)

    pothole_detected = False

    for det in results.xyxy[0]:
        class_id = det['class']
        class_name = model.names[class_id].strip().lower()  # Retrieve the class name for the current detection
        score = det['conf']

        if score > confidence_threshold:  # Apply confidence threshold
            if class_name == pothole_class_name:
                pothole_detected = True
                # Draw bounding box around detected pothole
                x1, y1, x2, y2 = det['box']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)  # Green bounding box
                text = f"{class_name}: {score:.2f}"
                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                # Move plotter to a specific location and wait for 2 seconds
                center_x = 100  # Example fixed coordinates
                center_y = 100
                threading.Thread(target=move_plotter_to_location, args=(ad, center_x, center_y)).start()

    if not pothole_detected:
        running.set()
    else:
        running.clear()

    # Display the frame
    cv2.imshow('Video', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

# Stop the random line drawing thread
running.clear()
line_thread.join()

# Disconnect from the Axidraw plotter
ad.disconnect()
