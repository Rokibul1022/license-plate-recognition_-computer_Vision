"""
Comprehensive Evaluation Script
Compares OCR, Zero-Shot VLM, and Fine-tuned VLM
"""

from evaluation.evaluator import Evaluator
from baselines.ocr_baseline import OCRBaseline
from vlm.trainer import VLMTrainer
import json
import os

# Load test data
print("Loading test data...")
with open('unified_dataset.json') as f:
    test_data = json.load(f)

# Take subset for evaluation
test_subset = test_data[:100]  # Adjust as needed

# Initialize models
print("\nInitializing models...")
ocr_model = OCRBaseline(use_easyocr=True)
zero_shot_vlm = VLMTrainer("google/paligemma-3b-pt-224")
finetuned_vlm = VLMTrainer()
finetuned_vlm.model = finetuned_vlm.model.from_pretrained("outputs/vlm/finetuned_model")

# Evaluate each method
evaluator = Evaluator()

print("\n" + "="*50)
print("Evaluating OCR Baseline...")
print("="*50)
ocr_predictions = []
for item in test_subset:
    try:
        pred = ocr_model.predict(item['image_path'])
        ocr_predictions.append(pred)
    except:
        ocr_predictions.append({'plate': '', 'color': '', 'type': ''})

ocr_results = evaluator.evaluate_method(ocr_predictions, test_subset, 'OCR')

print("\n" + "="*50)
print("Evaluating Zero-Shot VLM...")
print("="*50)
zeroshot_predictions = []
for item in test_subset:
    try:
        pred = zero_shot_vlm.predict(item['image_path'])
        zeroshot_predictions.append(pred)
    except:
        zeroshot_predictions.append({'plate': '', 'color': '', 'type': ''})

zeroshot_results = evaluator.evaluate_method(zeroshot_predictions, test_subset, 'Zero-Shot VLM')

print("\n" + "="*50)
print("Evaluating Fine-tuned VLM...")
print("="*50)
finetuned_predictions = []
for item in test_subset:
    try:
        pred = finetuned_vlm.predict(item['image_path'])
        finetuned_predictions.append(pred)
    except:
        finetuned_predictions.append({'plate': '', 'color': '', 'type': ''})

finetuned_results = evaluator.evaluate_method(finetuned_predictions, test_subset, 'Fine-tuned VLM')

# Compare methods
print("\n" + "="*50)
print("COMPARISON RESULTS")
print("="*50)

results_dict = {
    'OCR Baseline': ocr_results,
    'Zero-Shot VLM': zeroshot_results,
    'Fine-tuned VLM': finetuned_results
}

comparison = evaluator.compare_methods(results_dict)

# Print comparison table
print("\n{:<20} {:<15} {:<15} {:<15} {:<15}".format(
    "Method", "Plate Acc", "Char Acc", "Color Acc", "Type Acc"
))
print("-" * 80)
for method, metrics in comparison.items():
    print("{:<20} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format(
        method,
        metrics.get('plate_accuracy', 0),
        metrics.get('char_accuracy', 0),
        metrics.get('color_accuracy', 0),
        metrics.get('type_accuracy', 0)
    ))

print("\n" + "="*50)
print("Evaluation complete!")
print("Results saved to: outputs/evaluation/")
print("="*50)
