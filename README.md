# Ping Pong Deep Learning

Train a YOLOv11 model to detect ping pong ball position and game state (serve, missed shot, etc). Loosely based on [TTNet Research Paper](https://arxiv.org/pdf/2004.09927), using [OpenTTGames Dataset](https://lab.osai.ai/)

# Getting Started

1. Download dependencies and dataset

```bash
# Download dependencies
pip install -r requirements.txt

# Note: To enable GPU usage
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Download dataset
cd prepare_dataset
python download_dataset.py
python unzip.py
```

2. Extract images from videos and create a corresponding label.txt so YOLO can train

```bash
# Extract images
python prepare_dataset/extract_ball_images_and_labels.py

# Optional: Visualize dataset
python src/visualize_annotated_data.py
```

3. Train the model!

```bash
# Train model
python src/train.py
```

4. Use the model

```bash
python src/demo.py
```
