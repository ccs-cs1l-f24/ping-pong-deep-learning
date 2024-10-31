import json
import os
import cv2

# Set max images for testing purposes
MAX_IMAGES_PER_VIDEO = float('inf')

# Split test images into 50% validation and 50% test

train_json_files = [
    '../dataset/opentt/train/annotations/game_1/ball_markup.json',
    '../dataset/opentt/train/annotations/game_2/ball_markup.json',
    '../dataset/opentt/train/annotations/game_3/ball_markup.json',
    '../dataset/opentt/train/annotations/game_4/ball_markup.json',
    '../dataset/opentt/train/annotations/game_5/ball_markup.json'
]

train_video_files = [
    '../dataset/opentt/train/videos/game_1.mp4',
    '../dataset/opentt/train/videos/game_2.mp4',
    '../dataset/opentt/train/videos/game_3.mp4',
    '../dataset/opentt/train/videos/game_4.mp4',
    '../dataset/opentt/train/videos/game_5.mp4'
]

test_json_files = [
    '../dataset/opentt/test/annotations/test_1/ball_markup.json',
    '../dataset/opentt/test/annotations/test_2/ball_markup.json',
    '../dataset/opentt/test/annotations/test_3/ball_markup.json',
]

test_video_files = [
    '../dataset/opentt/test/videos/test_1.mp4',
    '../dataset/opentt/test/videos/test_2.mp4',
    '../dataset/opentt/test/videos/test_3.mp4',
]

val_json_files = [
    '../dataset/opentt/test/annotations/test_4/ball_markup.json',
    '../dataset/opentt/test/annotations/test_5/ball_markup.json',
    '../dataset/opentt/test/annotations/test_6/ball_markup.json',
    '../dataset/opentt/test/annotations/test_7/ball_markup.json',
]

val_video_files = [
    '../dataset/opentt/test/videos/test_4.mp4',
    '../dataset/opentt/test/videos/test_5.mp4',
    '../dataset/opentt/test/videos/test_6.mp4',
    '../dataset/opentt/test/videos/test_7.mp4',
]

# Create folders for images and labels if they don't exist
train_image_dir = '../dataset/images/train/'
train_label_dir = '../dataset/labels/train/'
test_image_dir = '../dataset/images/test'
test_label_dir = '../dataset/labels/test'
val_image_dir = '../dataset/images/val'
val_label_dir = '../dataset/labels/val'
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Bounding box width and height (in normalized coordinates, can change as needed)
bbox_width = 0.01
bbox_height = 0.02

# Dictionary to keep track of processed frames to avoid duplication
processed_frames = {}


def extract_ball_images_and_labels(json_files, video_files, image_output_dir, label_output_dir):
    # Iterate over each game
    for json_file, video_file in zip(json_files, video_files):
        # Extract game name from file path
        game_name = os.path.basename(video_file).split('.')[0]

        with open(json_file, 'r') as file:
            ball_positions = json.load(file)

        # Load the corresponding video
        cap = cv2.VideoCapture(video_file)

        # Assuming video resolution (adjust if necessary)
        image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Limit number of images while testing
        img_cnt = 0

        for frame_index, position in ball_positions.items():
            img_cnt += 1
            if img_cnt > MAX_IMAGES_PER_VIDEO:
                break

            frame_index = int(frame_index)

            # Check if the frame has already been processed
            if (video_file, frame_index) in processed_frames:
                continue

            # Mark this frame as processed
            processed_frames[(video_file, frame_index)] = True

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                # Save the frame as an image, prepending the game name
                image_path = os.path.join(
                    image_output_dir, f'{game_name}_frame_{frame_index}.jpg')
                if not os.path.exists(image_path):
                    cv2.imwrite(image_path, frame)

                # Get the coordinates of the ball
                x = position['x']
                y = position['y']

                # Normalize coordinates
                x_center = x / image_width
                y_center = y / image_height

                # Create label file, prepending the game name
                label_path = os.path.join(
                    label_output_dir, f'{game_name}_frame_{frame_index}.txt')
                with open(label_path, 'a') as f:
                    f.write(
                        f'0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n')

        cap.release()


extract_ball_images_and_labels(
    train_json_files, train_video_files, train_image_dir, train_label_dir)
extract_ball_images_and_labels(
    test_json_files, test_video_files, test_image_dir, test_label_dir)
extract_ball_images_and_labels(
    val_json_files, val_json_files, val_image_dir, val_label_dir)
