# Ping Pong Deep Learning

Train a YOLOv11 model to detect ping pong ball position and game state (serve, missed shot, etc). Loosely based on [TTNet Research Paper](https://arxiv.org/pdf/2004.09927), using [OpenTTGames Dataset](https://lab.osai.ai/)

# Getting Started

```
# Download dependencies
pip install -r requirements.txt

# Download dataset
python prepare_dataset/download_dataset.py

# Train model
python src/main.py
```
