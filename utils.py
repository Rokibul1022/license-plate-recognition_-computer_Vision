"""
Utility Functions for BD License Plate Recognition System
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def visualize_detection(image_path, detections, output_path=None):
    """Visualize detection results on image"""
    image = cv2.imread(image_path)
    
    for det in detections:
        bbox = det['plate_bbox']
        conf = det['confidence']
        
        # Draw bounding box
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Add confidence text
        text = f"Conf: {conf:.2f}"
        cv2.putText(image, text, (bbox[0], bbox[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, image)
    
    return image

def visualize_recognition(image, plate_text, color, vehicle_type, output_path=None):
    """Visualize recognition results"""
    if isinstance(image, str):
        image = cv2.imread(image)
    
    # Add text overlay
    text_lines = [
        f"Plate: {plate_text}",
        f"Color: {color}",
        f"Type: {vehicle_type}"
    ]
    
    y_offset = 30
    for line in text_lines:
        cv2.putText(image, line, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
    
    if output_path:
        cv2.imwrite(output_path, image)
    
    return image

def create_comparison_plot(results_dict, output_path='outputs/comparison.png'):
    """Create comparison plot for different methods"""
    methods = list(results_dict.keys())
    metrics = ['plate_accuracy', 'char_accuracy', 'color_accuracy', 'type_accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Method Comparison', fontsize=16)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [results_dict[m].get(metric, 0) for m in methods]
        
        bars = ax.bar(methods, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_ylabel('Accuracy')
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def save_results(results, output_file):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

def load_results(input_file):
    """Load results from JSON file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_metrics(predictions, ground_truth):
    """Calculate evaluation metrics"""
    metrics = {
        'total': len(predictions),
        'correct_plates': 0,
        'correct_colors': 0,
        'correct_types': 0,
        'char_accuracy': []
    }
    
    for pred, gt in zip(predictions, ground_truth):
        # Plate match
        if pred.get('plate', '').strip() == gt.get('plate', '').strip():
            metrics['correct_plates'] += 1
        
        # Color match
        if pred.get('color', '').lower() == gt.get('color', '').lower():
            metrics['correct_colors'] += 1
        
        # Type match
        if pred.get('type', '').lower() == gt.get('type', '').lower():
            metrics['correct_types'] += 1
        
        # Character accuracy
        pred_plate = pred.get('plate', '')
        gt_plate = gt.get('plate', '')
        if pred_plate and gt_plate:
            matches = sum(1 for p, g in zip(pred_plate, gt_plate) if p == g)
            char_acc = matches / max(len(pred_plate), len(gt_plate))
            metrics['char_accuracy'].append(char_acc)
    
    # Calculate percentages
    metrics['plate_accuracy'] = metrics['correct_plates'] / metrics['total']
    metrics['color_accuracy'] = metrics['correct_colors'] / metrics['total']
    metrics['type_accuracy'] = metrics['correct_types'] / metrics['total']
    metrics['avg_char_accuracy'] = np.mean(metrics['char_accuracy']) if metrics['char_accuracy'] else 0
    
    return metrics

def format_plate_text(text):
    """Format plate text for display"""
    # Remove extra spaces and normalize
    text = ' '.join(text.split())
    return text

def validate_bd_plate(plate_text):
    """Validate Bangladeshi plate format"""
    import re
    
    # Basic validation
    if not plate_text or len(plate_text) < 5:
        return False
    
    # Check for required components
    has_bengali = any('\u0980' <= c <= '\u09FF' for c in plate_text)
    has_numbers = any(c.isdigit() for c in plate_text)
    
    return has_bengali and has_numbers

def create_confusion_matrix(predictions, ground_truth, labels, output_path='outputs/confusion_matrix.png'):
    """Create confusion matrix visualization"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def log_experiment(config, results, output_dir='outputs/experiments'):
    """Log experiment configuration and results"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'experiment_{timestamp}.json')
    
    experiment_log = {
        'timestamp': timestamp,
        'config': config,
        'results': results
    }
    
    with open(log_file, 'w') as f:
        json.dump(experiment_log, f, indent=2)
    
    print(f"Experiment logged to {log_file}")
    return log_file

def print_summary(results):
    """Print formatted summary of results"""
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  Plate Accuracy:     {metrics.get('plate_accuracy', 0):.3f}")
        print(f"  Character Accuracy: {metrics.get('char_accuracy', 0):.3f}")
        print(f"  Color Accuracy:     {metrics.get('color_accuracy', 0):.3f}")
        print(f"  Type Accuracy:      {metrics.get('type_accuracy', 0):.3f}")
    
    print("\n" + "="*60)
