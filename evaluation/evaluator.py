import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu
import json

class Evaluator:
    def __init__(self):
        self.results = {
            'ocr': {'plate_acc': [], 'char_acc': []},
            'zero_shot': {'plate_acc': [], 'char_acc': [], 'color_acc': [], 'type_acc': []},
            'finetuned': {'plate_acc': [], 'char_acc': [], 'color_acc': [], 'type_acc': []}
        }
    
    def char_accuracy(self, pred, gt):
        """Character-level accuracy"""
        if not pred or not gt:
            return 0.0
        
        matches = sum(1 for p, g in zip(pred, gt) if p == g)
        return matches / max(len(pred), len(gt))
    
    def exact_match(self, pred, gt):
        """Exact plate match"""
        return 1.0 if pred.strip() == gt.strip() else 0.0
    
    def bleu_score(self, pred, gt):
        """BLEU score for text similarity"""
        reference = [gt.split()]
        candidate = pred.split()
        return sentence_bleu(reference, candidate)
    
    def evaluate_method(self, predictions, ground_truth, method_name):
        """Evaluate a single method"""
        plate_accs = []
        char_accs = []
        color_accs = []
        type_accs = []
        
        for pred, gt in zip(predictions, ground_truth):
            # Plate accuracy
            plate_accs.append(self.exact_match(pred.get('plate', ''), gt.get('plate', '')))
            char_accs.append(self.char_accuracy(pred.get('plate', ''), gt.get('plate', '')))
            
            # Attribute accuracy (if available)
            if 'color' in pred and 'color' in gt:
                color_accs.append(1.0 if pred['color'] == gt['color'] else 0.0)
            
            if 'type' in pred and 'type' in gt:
                type_accs.append(1.0 if pred['type'] == gt['type'] else 0.0)
        
        results = {
            'plate_accuracy': np.mean(plate_accs) if plate_accs else 0,
            'char_accuracy': np.mean(char_accs) if char_accs else 0,
            'color_accuracy': np.mean(color_accs) if color_accs else 0,
            'type_accuracy': np.mean(type_accs) if type_accs else 0
        }
        
        return results
    
    def compare_methods(self, results_dict, output_dir="outputs/evaluation"):
        """Compare all methods and generate plots"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Comparison table
        comparison = {}
        for method, metrics in results_dict.items():
            comparison[method] = metrics
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        methods = list(comparison.keys())
        metrics = ['plate_accuracy', 'char_accuracy', 'color_accuracy', 'type_accuracy']
        titles = ['Plate Accuracy', 'Character Accuracy', 'Color Accuracy', 'Type Accuracy']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            values = [comparison[m].get(metric, 0) for m in methods]
            ax.bar(methods, values)
            ax.set_title(title)
            ax.set_ylabel('Accuracy')
            ax.set_ylim([0, 1])
            
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/method_comparison.png", dpi=300)
        plt.close()
        
        # Save results
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Evaluation results saved to {output_dir}")
        return comparison
    
    def robustness_evaluation(self, model, test_sets, output_dir="outputs/evaluation"):
        """Evaluate robustness on different conditions"""
        conditions = ['blur', 'night', 'angle']
        results = {}
        
        for condition in conditions:
            if condition in test_sets:
                preds = [model.predict(img) for img in test_sets[condition]['images']]
                gt = test_sets[condition]['labels']
                results[condition] = self.evaluate_method(preds, gt, condition)
        
        # Plot robustness
        fig, ax = plt.subplots(figsize=(10, 6))
        
        conditions_present = list(results.keys())
        plate_accs = [results[c]['plate_accuracy'] for c in conditions_present]
        
        ax.bar(conditions_present, plate_accs)
        ax.set_title('Robustness Evaluation')
        ax.set_ylabel('Plate Accuracy')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/robustness.png", dpi=300)
        plt.close()
        
        return results
