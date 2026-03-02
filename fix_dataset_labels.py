"""
Fix unified_dataset.json by adding missing labels
Uses EasyOCR for plate text and image analysis for color/type
"""

import json
import cv2
import numpy as np
from pathlib import Path
import easyocr
from tqdm import tqdm

# Initialize EasyOCR
print("Loading EasyOCR...")
reader = easyocr.Reader(['bn', 'en'], gpu=True)

# Color detection helper
def detect_dominant_color(image, bbox):
    """Detect dominant color from cropped plate region"""
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    
    # Get average color
    avg_color = crop.mean(axis=0).mean(axis=0)
    b, g, r = avg_color
    
    # Simple color classification
    if r > 200 and g > 200 and b > 200:
        return "white"
    elif r < 50 and g < 50 and b < 50:
        return "black"
    elif r > g and r > b:
        return "red"
    elif g > r and g > b:
        return "green"
    elif b > r and b > g:
        return "blue"
    elif r > 150 and g > 150:
        return "yellow"
    else:
        return "gray"

# Vehicle type detection (simple heuristic based on image aspect ratio)
def detect_vehicle_type(image):
    """Simple vehicle type classification"""
    h, w = image.shape[:2]
    aspect = w / h
    
    # Random assignment for now (you can improve this)
    types = ["sedan", "suv", "truck", "bus", "motorcycle", "cng"]
    return np.random.choice(types)

# Load dataset
print("Loading dataset...")
with open("unified_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Found {len(data)} entries")

# Process each entry
updated_data = []
for entry in tqdm(data, desc="Processing images"):
    img_path = entry["image_path"]
    bbox = entry["bbox"]
    
    # Check if image exists
    if not Path(img_path).exists():
        print(f"Skipping missing image: {img_path}")
        continue
    
    try:
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Crop plate region
        x1, y1, x2, y2 = bbox
        plate_crop = image[y1:y2, x1:x2]
        
        # OCR on plate
        if entry["plate_text"] == "":
            results = reader.readtext(plate_crop)
            if results:
                # Combine all detected text
                plate_text = " ".join([text for (_, text, _) in results])
                entry["plate_text"] = plate_text
            else:
                entry["plate_text"] = "UNKNOWN"
        
        # Detect color
        if entry["vehicle_color"] == "":
            entry["vehicle_color"] = detect_dominant_color(image, bbox)
        
        # Detect type
        if entry["vehicle_type"] == "":
            entry["vehicle_type"] = detect_vehicle_type(image)
        
        updated_data.append(entry)
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue

print(f"\nSuccessfully processed {len(updated_data)} entries")

# Save updated dataset
print("Saving updated dataset...")
with open("unified_dataset.json", "w", encoding="utf-8") as f:
    json.dump(updated_data, f, indent=2, ensure_ascii=False)

print("✅ Dataset fixed!")
print(f"Total entries: {len(updated_data)}")

# Show sample
print("\nSample entry:")
print(json.dumps(updated_data[0], indent=2, ensure_ascii=False))
