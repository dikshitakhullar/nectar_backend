from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
# model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' for better accuracy but slower training
# Load the last checkpoint
model = YOLO('runs/detect/yolov8_home_decor72/weights/last.pt')  # Path to the last saved weights


# Train the model on your custom dataset
model.train(
    data='dataset/dataset.yaml',  # replace with local directly on dataset files
    epochs=50,                     # Number of training epochs
    imgsz=640,                     # Image size
    batch=16,                      # Batch size
    name='yolov8_home_decor8',      # Experiment name
    pretrained=True                # Use pre-trained weights
)

print("Training complete. Best model saved at 'runs/detect/yolov8_home_decor/weights/best.pt'")
