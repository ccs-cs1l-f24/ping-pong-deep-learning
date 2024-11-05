from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("best.pt")

# Define the video path and capture it
video_path = "../dataset/opentt/test/videos/test_1.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process each frame in the video
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or error encountered.")
        break

    # Make a copy of the frame for YOLO predictions
    yolo_frame = frame.copy()

    # YOLO model prediction
    results = model.predict(source=frame, conf=0.25, verbose=False)
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            # Draw YOLO predicted bounding box (in blue)
            cv2.rectangle(yolo_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Display the frame with YOLO predictions
    cv2.imshow("YOLO Predictions", yolo_frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close display windows
cap.release()
cv2.destroyAllWindows()
