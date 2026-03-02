"""
YOLO Training Script for License Plate Detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.detector import PlateDetector
import yaml

if __name__ == '__main__':
    # Create data.yaml for YOLO
    data_config = {
        'path': '../data',
        'train': 'Bangladeshi_Vehicle_Image/Bangladeshi_Vehicle_Image_with_visible_license_plate',
        'val': 'Bangladeshi_Vehicle_Image/Bangladeshi_Vehicle_Image_with_visible_license_plate',
        'nc': 1,
        'names': ['license_plate']
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_config, f)
    
    # Train detector
    detector = PlateDetector('yolov8n.pt')
    
    print("Starting YOLO training...")
    print("This will take several hours depending on your GPU")
    
    results = detector.train(
        data_yaml='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16
    )
    
    print("Training complete!")
    print(f"Model saved to: outputs/detection/plate_detector/weights/best.pt")
