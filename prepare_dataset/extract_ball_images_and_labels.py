import json
import os
import cv2

MAX_IMAGES_PER_VIDEO = 50

# List of JSON files for different games
json_files = [
    '../dataset/opentt/training/annotations/game_1/ball_markup.json',
    '../dataset/opentt/training/annotations/game_2/ball_markup.json',
    '../dataset/opentt/training/annotations/game_3/ball_markup.json',
    '../dataset/opentt/training/annotations/game_4/ball_markup.json',
    '../dataset/opentt/training/annotations/game_5/ball_markup.json'
]

# List of corresponding video files for different games
video_files = [
    '../dataset/opentt/training/videos/game_1.mp4',
    '../dataset/opentt/training/videos/game_2.mp4',
    '../dataset/opentt/training/videos/game_3.mp4',
    '../dataset/opentt/training/videos/game_4.mp4',
    '../dataset/opentt/training/videos/game_5.mp4'
]

# Create folders for images and labels if they don't exist
image_dir = '../dataset/images/train/'
label_dir = '../dataset/labels/train/'
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# Bounding box width and height (in normalized coordinates, adjustable based on ball size)
bbox_width = 0.01  # You may need to adjust this value for better results
bbox_height = 0.02

# Dictionary to keep track of processed frames to avoid duplication
processed_frames = {}

# Iterate over each game
for json_file, video_file in zip(json_files, video_files):
    # Extract game name from file path
    game_name = os.path.basename(video_file).split('.')[0]

    # Load the JSON data
    with open(json_file, 'r') as file:
        ball_positions = json.load(file)

    # Load the corresponding video
    cap = cv2.VideoCapture(video_file)

    # Assuming video resolution (adjust if necessary)
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    img_cnt = 0    

    for frame_index, position in ball_positions.items():
        img_cnt += 1
        if img_cnt > MAX_IMAGES_PER_VIDEO:
            break

        frame_index = int(frame_index)  # Make sure it's an integer

        # Check if the frame has already been processed
        if (video_file, frame_index) in processed_frames:
            continue
        
        # Mark this frame as processed
        processed_frames[(video_file, frame_index)] = True

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            # Save the frame as an image, prepending the game name
            image_path = os.path.join(image_dir, f'{game_name}_frame_{frame_index}.jpg')
            if not os.path.exists(image_path):
                cv2.imwrite(image_path, frame)

            # Get the coordinates of the ball
            x = position['x']
            y = position['y']

            # Normalize coordinates
            x_center = x / image_width
            y_center = y / image_height

            # Create label file, prepending the game name
            label_path = os.path.join(label_dir, f'{game_name}_frame_{frame_index}.txt')
            with open(label_path, 'a') as f:
                f.write(f'0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n')

    cap.release()
