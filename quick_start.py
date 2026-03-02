"""
Quick Start Script - Runs Initial Setup
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print('='*60)
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False
    return True

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  BD License Plate Recognition System - Quick Start      ║
    ║  Optimized for RTX 3060 12GB                            ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Create directories
    dirs = ['outputs', 'outputs/detection', 'outputs/vlm', 'outputs/evaluation', 
            'outputs/frames', 'database', 'models']
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("✅ Directory structure created")
    
    # Step 1: Unify datasets
    if run_command(f"{sys.executable} data_unification.py", "Dataset Unification"):
        pass
    
    # Step 2: Generate database
    if run_command(f"{sys.executable} database/generate_db.py", "Synthetic Database Generation"):
        pass
    
    print(f"""
    
    ╔══════════════════════════════════════════════════════════╗
    ║  Initial Setup Complete!                                 ║
    ╚══════════════════════════════════════════════════════════╝
    
    Next Steps:
    
    1. Train YOLO Detector (requires GPU):
       python detection/train_yolo.py
    
    2. Fine-tune VLM with QLoRA (requires GPU):
       python vlm/train_vlm.py
    
    3. Run Evaluation:
       python evaluation/run_evaluation.py
    
    4. Launch Demo App:
       streamlit run app/streamlit_app.py
    
    For detailed instructions, see README.md
    """)

if __name__ == "__main__":
    main()
