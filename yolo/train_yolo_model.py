from ultralytics import YOLO
model = YOLO("yolov8n")  # This loads the YOLOv8 Nano model

results = model.train(data="/data/yolo_combined_dataset/data.yaml",
                      epochs=50,
                      imgsz=256,
                      batch=8,
                      workers=4)