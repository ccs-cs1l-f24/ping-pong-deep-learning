from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="../dataset/opentt.yaml", 
        project="..", 
        epochs=300, 
        patience=10, 
        batch=-1, 
        imgsz=640
    )
