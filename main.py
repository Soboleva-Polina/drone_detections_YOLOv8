from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data=r'C:\Users\User\Desktop\drone_yoloV9\config.yaml', epochs=100)