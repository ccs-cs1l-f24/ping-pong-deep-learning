from ultralytics import YOLO
import cv2
import os

# Load the YOLO model
model = YOLO("best.pt")

# Define directories
images_dir = "../dataset/images/test"
labels_dir = "../dataset/labels/test"

# Iterate over all images in the directory
for image_filename in os.listdir(images_dir):
    if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    # Load image
    image_path = os.path.join(images_dir, image_filename)
    image = cv2.imread(image_path)
    
    # Make a copy of the image for YOLO predictions
    yolo_image = image.copy()

    # YOLO model prediction
    results = model.predict(source=image_path, conf=0.25, verbose=False)
    for result in results:
        for box in result.boxes:
            print(box)
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            # Draw YOLO predicted bounding box (in blue)
            cv2.rectangle(yolo_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Load label
    label_filename = image_filename.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_filename)

    if not os.path.exists(label_path):
        print(f"Label file not found for {image_filename}, skipping...")
        continue

    # Draw ground truth bounding boxes
    with open(label_path, 'r') as label_file:
        for line in label_file:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Malformed label in {label_filename}, skipping...")
                continue

            class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)
            img_height, img_width, _ = image.shape

            # Denormalize to pixel coordinates
            x_center_pixel = int(x_center * img_width)
            y_center_pixel = int(y_center * img_height)
            bbox_width_pixel = int(bbox_width * img_width)
            bbox_height_pixel = int(bbox_height * img_height)

            # Calculate top-left and bottom-right points of the bounding box
            top_left = (x_center_pixel - bbox_width_pixel // 2, y_center_pixel - bbox_height_pixel // 2)
            bottom_right = (x_center_pixel + bbox_width_pixel // 2, y_center_pixel + bbox_height_pixel // 2)

            # Draw the bounding box on the image (in green)
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Display YOLO predictions
    cv2.imshow("YOLO Predictions", yolo_image)
    # Display Ground Truth
    cv2.imshow("Ground Truth", image)
    
    # Wait for a key press
    key = cv2.waitKey(30)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
