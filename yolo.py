
import cv2
import torch


model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',
    pretrained=True
)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)

    
    detections = results.pandas().xyxy[0]

    label_count = {}

    for _, row in detections.iterrows():
        label = row['name']
        confidence = row['confidence']

      
        label_count[label] = label_count.get(label, 0) + 1

        
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        
        text = f"{label} {confidence:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

   
    print("Detected Objects:", label_count)

    
    cv2.imshow("Object Detection - YOLO", frame)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()

