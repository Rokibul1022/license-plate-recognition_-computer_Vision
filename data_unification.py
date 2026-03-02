import os
import json
import glob
from pathlib import Path
from PIL import Image

def unify_datasets(data_root="data", output_file="unified_dataset.json"):
    """Unify all three datasets into a single format"""
    unified_data = []
    
    # Dataset 1: Bangladeshi_Vehicle_Image (YOLO format)
    vehicle_dir = Path(data_root) / "Bangladeshi_Vehicle_Image/Bangladeshi_Vehicle_Image_with_visible_license_plate"
    if vehicle_dir.exists():
        for img_file in vehicle_dir.glob("*.jpg"):
            txt_file = img_file.with_suffix('.txt')
            if txt_file.exists():
                with open(txt_file) as f:
                    line = f.readline().strip().split()
                    if len(line) >= 5:
                        _, x_center, y_center, width, height = map(float, line[:5])
                        img = Image.open(img_file)
                        w, h = img.size
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        unified_data.append({
                            "image_path": str(img_file),
                            "plate_text": "",  # Not available
                            "vehicle_color": "",  # Not available
                            "vehicle_type": "",  # Not available
                            "bbox": [x1, y1, x2, y2]
                        })
    
    print(f"Processed {len(unified_data)} samples")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, indent=2, ensure_ascii=False)
    
    return unified_data

if __name__ == "__main__":
    data = unify_datasets()
    print(f"Total unified samples: {len(data)}")
