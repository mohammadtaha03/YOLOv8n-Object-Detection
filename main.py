from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,
    imgsz=640
)

for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    label = model.names[class_id]
    print(f"{label}: {confidence:.2f}")

print("Detection completed!")
