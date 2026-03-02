import sys
import os
sys.path.append('.')

from vlm.trocr_trainer import TrOCRTrainer
from vlm.attribute_classifier import AttributeTrainer

if __name__ == '__main__':
    # Create output directory
    os.makedirs('outputs/trocr', exist_ok=True)
    
    print("="*60)
    print("PHASE 1: Training TrOCR for Plate Text Recognition")
    print("="*60)
    trocr_trainer = TrOCRTrainer()
    trocr_trainer.train('unified_dataset.json', epochs=10, batch_size=8, lr=5e-5)
    print("\nTrOCR training complete!")
    
    print("\n" + "="*60)
    print("PHASE 2: Training Attribute Classifier (Color + Type)")
    print("="*60)
    attr_trainer = AttributeTrainer()
    attr_trainer.train('unified_dataset.json', epochs=10, batch_size=32, lr=1e-3)
    print("\nAttribute classifier training complete!")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("Models saved to:")
    print("  - outputs/trocr/plate_reader/")
    print("  - outputs/trocr/attribute_classifier.pth")
