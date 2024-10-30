from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data="../dataset/opentt.yaml", epochs=3, imgsz=640) # TODO: Play with epochs and imgsz




"""
TODO: Fix the datasets dir
- put the downloaded videos somewhere like opentt/original
- put the images and labels dir in opentt
- put the yaml file in the right spot and make it reference opentt
- AHHHHH


"""