import os
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best.pt")

# Define video path
video_path = "../dataset/opentt/test/videos/test_1.mp4"

# Open video capture
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    exit()

# Loop through each frame in the video
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or error reading frame.")
        break

    # Make a copy of the frame for YOLO predictions
    yolo_frame = frame.copy()

    # YOLO model prediction
    results = model.predict(source=frame, conf=0.25, verbose=False)
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            # Draw YOLO predicted bounding box (in blue)
            cv2.rectangle(yolo_frame, (x_min, y_min),
                          (x_max, y_max), (255, 0, 0), 2)

    # Display YOLO predictions
    cv2.imshow("YOLO Predictions", yolo_frame)

    # Wait for a key press
    key = cv2.waitKey(30)

    if key == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
