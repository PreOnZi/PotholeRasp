import os
import cv2
import cairosvg
import numpy as np
from ultralytics import YOLO

# Function to convert SVG to PNG and then to OpenCV format
def svg_to_image(svg_path):
    png_data = cairosvg.svg2png(url=svg_path)
    image = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_UNCHANGED)
    return image

# Define the absolute path to the videos directory
VIDEOS_DIR = '/Users/ondrejzika/Desktop/potholes/YOLO/VIDEOS_DIR'

# Define the absolute path to the input video file
video_path = os.path.join(VIDEOS_DIR, '03_1.mp4')
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
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Define the path to your custom model weights file
custom_model_path = '/Users/ondrejzika/Desktop/Pothole01/pothole_dataset_v8/runs/detect/yolov8n_v8_50e11/weights/best.pt'

# Load the custom model
model = YOLO(custom_model_path)

threshold = 0.5

# Paths to SVG plotter images
x_svg_path = '/Users/ondrejzika/Desktop/potholes/YOLO/x.svg'
circle_svg_path = '/Users/ondrejzika/Desktop/potholes/YOLO/circle.svg'

# Convert SVG to images
x_img = svg_to_image(x_svg_path)
circle_img = svg_to_image(circle_svg_path)

if x_img is None or circle_img is None:
    raise FileNotFoundError("Plotter images not found. Make sure x.svg and circle.svg are in the correct path.")

while ret:
    results = model(frame)
    pothole_detected_in_frame = False

    for result in results.xyxy[0]:
        x1, y1, x2, y2, score, class_id = result
        score = float(score)

        if score > threshold:
            pothole_detected_in_frame = True
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text = f"{results.names[int(class_id)]}: {score:.2f}"
            cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Determine which plotter image to show
    if pothole_detected_in_frame:
        plotter_img = circle_img
        print('circle.svg')
    else:
        plotter_img = x_img
        print('x.svg')

    # Display the plotter image
    cv2.imshow('Plotter', plotter_img)
    # Display the current frame with bounding boxes
    cv2.imshow('Frame', frame)

    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
