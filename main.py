from ultralytics import YOLO

# Load the pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Run object detection on the input image
results = model.predict(
    source="images/bus.jpg",
    save=True,
    imgsz=640
)

# Print each detected object with its confidence score
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    label = model.names[class_id]
    print(f"{label}: {confidence:.2f}")

print("Detection completed!")
