import easyocr
import pytesseract
import cv2
import numpy as np

class OCRBaseline:
    def __init__(self, use_easyocr=True):
        self.use_easyocr = use_easyocr
        if use_easyocr:
            self.reader = easyocr.Reader(['bn', 'en'], gpu=True)
    
    def preprocess(self, image):
        """Preprocess plate image for OCR"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Threshold
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def recognize(self, image):
        """Recognize text from plate image"""
        preprocessed = self.preprocess(image)
        
        if self.use_easyocr:
            results = self.reader.readtext(preprocessed)
            text = ' '.join([r[1] for r in results])
        else:
            text = pytesseract.image_to_string(preprocessed, lang='ben+eng')
        
        return text.strip()
    
    def predict(self, image_path):
        """Full pipeline prediction"""
        image = cv2.imread(image_path)
        text = self.recognize(image)
        
        return {
            "plate": text,
            "color": "",  # Not available in OCR
            "type": ""    # Not available in OCR
        }
