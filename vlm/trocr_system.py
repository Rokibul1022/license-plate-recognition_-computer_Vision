import torch
from vlm.trocr_trainer import TrOCRTrainer
from vlm.attribute_classifier import AttributeTrainer

class TrOCRSystem:
    def __init__(self, trocr_path="outputs/trocr/plate_reader", classifier_path="outputs/trocr/attribute_classifier.pth"):
        # Load TrOCR for text
        self.trocr = TrOCRTrainer()
        self.trocr.processor = torch.load(f"{trocr_path}/processor_config.json") if False else self.trocr.processor
        from transformers import VisionEncoderDecoderModel, TrOCRProcessor
        self.trocr.model = VisionEncoderDecoderModel.from_pretrained(trocr_path)
        self.trocr.processor = TrOCRProcessor.from_pretrained(trocr_path)
        self.trocr.model = self.trocr.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.trocr.model.eval()
        
        # Load attribute classifier
        self.classifier = AttributeTrainer()
        self.classifier.load(classifier_path)
    
    def predict(self, image):
        # Get plate text from TrOCR
        plate_text = self.trocr.predict(image)
        
        # Get attributes from classifier
        attributes = self.classifier.predict(image)
        
        return {
            'plate': plate_text,
            'color': attributes['color'],
            'type': attributes['type']
        }
