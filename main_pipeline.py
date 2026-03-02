"""
Main Training Pipeline for BD License Plate Recognition System
Executes all phases sequentially
"""

import os
import sys
from pathlib import Path

# Phase 1: Dataset Unification
print("="*50)
print("PHASE 1: Dataset Unification")
print("="*50)
from data_unification import unify_datasets
unified_data = unify_datasets()

# Phase 2: Generate Synthetic Database
print("\n" + "="*50)
print("PHASE 2: Synthetic Database Generation")
print("="*50)
from database.generate_db import create_database
os.makedirs("database", exist_ok=True)
create_database(num_records=1000)

# Phase 3: YOLO Training (Detection)
print("\n" + "="*50)
print("PHASE 3: YOLO Detection Training")
print("="*50)
print("Note: Prepare data.yaml for YOLO training")
print("Run: python detection/train_yolo.py")

# Phase 4: OCR Baseline
print("\n" + "="*50)
print("PHASE 4: OCR Baseline Setup")
print("="*50)
print("OCR baseline ready. No training required.")

# Phase 5: Zero-Shot VLM
print("\n" + "="*50)
print("PHASE 5: Zero-Shot VLM Evaluation")
print("="*50)
print("Zero-shot VLM ready for evaluation")

# Phase 6: VLM Fine-tuning
print("\n" + "="*50)
print("PHASE 6: VLM Fine-tuning with QLoRA")
print("="*50)
print("Note: Requires annotated data with plate text, color, and type")
print("Run: python vlm/train_vlm.py")

# Phase 7: Evaluation
print("\n" + "="*50)
print("PHASE 7: Evaluation & Comparison")
print("="*50)
print("Run: python evaluation/run_evaluation.py")

# Phase 8: Demo Application
print("\n" + "="*50)
print("PHASE 8: Launch Demo Application")
print("="*50)
print("Run: streamlit run app/streamlit_app.py")

print("\n" + "="*50)
print("SETUP COMPLETE")
print("="*50)
print("\nNext Steps:")
print("1. Prepare YOLO dataset (data.yaml)")
print("2. Train YOLO detector")
print("3. Annotate data for VLM training")
print("4. Fine-tune VLM with QLoRA")
print("5. Run evaluation")
print("6. Launch demo app")
