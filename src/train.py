from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data="../dataset/opentt.yaml", project="..", epochs=3, imgsz=640) # TODO: Play with epochs and imgsz