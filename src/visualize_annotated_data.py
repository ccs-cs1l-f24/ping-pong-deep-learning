import cv2
import os

images_dir = "../dataset/images/train"
labels_dir = "../dataset/labels/train"

for image_filename in os.listdir(images_dir):
    if not image_filename.endswith('.jpg'):
        continue

    # Load image
    image_path = os.path.join(images_dir, image_filename)
    image = cv2.imread(image_path)

    # Load label
    label_filename = image_filename.replace('.jpg', '.txt')
    label_path = os.path.join(labels_dir, label_filename)

    if not os.path.exists(label_path):
        continue

    with open(label_path, 'r') as label_file:
        for line in label_file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            bbox_width = float(parts[3])
            bbox_height = float(parts[4])

            # Denormalize coordinates to image size
            img_height, img_width, _ = image.shape
            x_center_pixel = int(x_center * img_width)
            y_center_pixel = int(y_center * img_height)
            bbox_width_pixel = int(bbox_width * img_width)
            bbox_height_pixel = int(bbox_height * img_height)

            # Calculate top-left and bottom-right points of the bounding box
            top_left = (x_center_pixel - bbox_width_pixel // 2, y_center_pixel - bbox_height_pixel // 2)
            bottom_right = (x_center_pixel + bbox_width_pixel // 2, y_center_pixel + bbox_height_pixel // 2)

            # Draw the bounding box on the image
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imshow("Image with Bounding Box", image)
    key = cv2.waitKey(0)

    if key == ord('q'):
        break

cv2.destroyAllWindows()