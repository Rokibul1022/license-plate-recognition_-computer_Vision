from ultralytics import YOLO
import cv2
import torch

class PlateDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        
    def train(self, data_yaml, epochs=50, imgsz=640, batch=16):
        """Train YOLO model"""
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=0 if torch.cuda.is_available() else 'cpu',
            project='outputs/detection',
            name='plate_detector'
        )
        return results
    
    def detect(self, image_path, conf=0.25):
        """Detect vehicles and plates"""
        results = self.model(image_path, conf=conf)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    "vehicle_bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "plate_bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "class": cls
                })
        
        return detections
    
    def crop_plate(self, image, bbox):
        """Crop plate region from image"""
        x1, y1, x2, y2 = bbox
        if isinstance(image, str):
            image = cv2.imread(image)
        return image[y1:y2, x1:x2]
