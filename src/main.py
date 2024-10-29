from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data="dataset/open-tt.yaml", epochs=100, imgsz=640) # TODO: Play with epochs and imgsz