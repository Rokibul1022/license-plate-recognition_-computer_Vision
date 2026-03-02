# 🚀 Step-by-Step Execution Guide

## Complete Workflow for BD License Plate Recognition System

---

## ⚙️ Prerequisites

### 1. System Requirements
- ✅ NVIDIA GPU (RTX 3060 12GB or better)
- ✅ CUDA 11.8+ installed
- ✅ Python 3.8+
- ✅ 16GB RAM minimum
- ✅ 50GB free disk space

### 2. Verify GPU
```bash
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📦 Step 1: Environment Setup (5 minutes)

```bash
# Navigate to project
cd Plate_recogniton_system

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, transformers, ultralytics; print('✅ All packages installed')"
```

---

## 🗂️ Step 2: Initial Setup (10 minutes)

```bash
# Run quick start script
python quick_start.py
```

This will:
- ✅ Create directory structure
- ✅ Unify datasets
- ✅ Generate synthetic database (1000 records)

**Expected Output:**
```
✅ Dataset unified: ~11,900 samples
✅ Database created: database/vehicle_registry.db
✅ 1000 synthetic vehicle records generated
```

---

## 🎯 Step 3: Train YOLO Detector (2-4 hours)

### Prepare YOLO Dataset

The system uses YOLO format annotations. Verify your data structure:

```
data/
└── Bangladeshi_Vehicle_Image/
    └── Bangladeshi_Vehicle_Image_with_visible_license_plate/
        ├── Vehicle1.jpg
        ├── Vehicle1.txt  (YOLO format: class x_center y_center width height)
        ├── Vehicle2.jpg
        ├── Vehicle2.txt
        ...
```

### Train

```bash
python detection/train_yolo.py
```

**Training Configuration:**
- Model: YOLOv8n (Nano)
- Epochs: 50
- Batch: 16
- Image size: 640x640

**Expected Output:**
```
Epoch 50/50: mAP@0.5: 0.85+
Model saved: outputs/detection/plate_detector/weights/best.pt
```

**Memory Usage:** ~6-8GB VRAM

---

## 🧠 Step 4: Fine-tune VLM with QLoRA (4-6 hours)

### Prerequisites
- Annotated data with plate text, color, and type
- Unified dataset JSON file

### Train

```bash
python vlm/train_vlm.py
```

**Training Configuration:**
- Model: PaliGemma 3B
- Quantization: 4-bit (NF4)
- LoRA rank: 16, alpha: 32
- Batch size: 2
- Epochs: 5
- Learning rate: 2e-4

**Expected Output:**
```
Trainable params: 8,388,608 || all params: 2,923,454,464 || trainable%: 0.2869
Epoch 1/5 Loss: 2.3456
Epoch 2/5 Loss: 1.8234
Epoch 3/5 Loss: 1.4567
Epoch 4/5 Loss: 1.2345
Epoch 5/5 Loss: 1.0987
Model saved: outputs/vlm/finetuned_model
```

**Memory Usage:** ~10-11GB VRAM

**⚠️ Important:** If you encounter OOM errors:
```python
# In vlm/train_vlm.py, reduce batch_size:
batch_size=1  # Instead of 2
```

---

## 📊 Step 5: Evaluation & Comparison (30 minutes)

```bash
python evaluation/run_evaluation.py
```

This will:
1. ✅ Evaluate OCR Baseline (EasyOCR)
2. ✅ Evaluate Zero-Shot VLM
3. ✅ Evaluate Fine-tuned VLM
4. ✅ Generate comparison plots
5. ✅ Save results to JSON

**Expected Output:**
```
Method              Plate Acc       Char Acc        Color Acc       Type Acc
--------------------------------------------------------------------------------
OCR Baseline        0.650           0.750           0.000           0.000
Zero-Shot VLM       0.700           0.800           0.600           0.550
Fine-tuned VLM      0.850           0.900           0.800           0.750

Results saved to: outputs/evaluation/
```

**Generated Files:**
- `outputs/evaluation/results.json`
- `outputs/evaluation/method_comparison.png`
- `outputs/evaluation/robustness.png`

---

## 🎨 Step 6: Launch Demo Application (Instant)

```bash
streamlit run app/streamlit_app.py
```

**Access:** http://localhost:8501

### Features:

#### 1. Image Upload Mode
- Upload vehicle image
- Automatic detection & recognition
- Verification against database
- Information retrieval options

#### 2. Video Upload Mode
- Process entire video
- Extract frames at 5 FPS
- Detect all vehicles
- Flag suspicious cases

#### 3. Database Query Mode
- Search by plate number
- View owner information
- Check movement history
- Verify registration status

---

## 🧪 Step 7: Test Individual Components

### Test Detection
```python
from detection.detector import PlateDetector

detector = PlateDetector('outputs/detection/plate_detector/weights/best.pt')
detections = detector.detect('test_image.jpg')
print(detections)
```

### Test Recognition
```python
from vlm.trainer import VLMTrainer

recognizer = VLMTrainer()
result = recognizer.predict('plate_crop.jpg')
print(result)
```

### Test Database Query
```python
from database.registry import VehicleRegistry

registry = VehicleRegistry()
info = registry.query('Dhaka মেট্রো-গ 12-3456')
print(info)
```

### Test Verification
```python
from verification.engine import VerificationEngine
from database.registry import VehicleRegistry

registry = VehicleRegistry()
verifier = VerificationEngine(registry)

result = verifier.verify(
    detected_plate='Dhaka মেট্রো-গ 12-3456',
    detected_color='White',
    detected_type='Sedan'
)
print(result)
```

---

## 📹 Step 8: Process Video

```python
from video_pipeline.processor import VideoProcessor
from detection.detector import PlateDetector
from vlm.trainer import VLMTrainer
from database.registry import VehicleRegistry
from verification.engine import VerificationEngine

# Initialize
detector = PlateDetector('outputs/detection/plate_detector/weights/best.pt')
recognizer = VLMTrainer()
registry = VehicleRegistry()
verifier = VerificationEngine(registry)

# Process
processor = VideoProcessor(detector, recognizer, verifier)
results = processor.process_video('test_video.mp4')

print(f"Processed {len(results)} detections")
```

---

## 🔍 Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1:** Reduce batch size
```python
# In training scripts
batch_size = 1  # Instead of 2 or 4
```

**Solution 2:** Use gradient accumulation
```python
# In vlm/trainer.py
accumulation_steps = 4
```

**Solution 3:** Clear cache
```python
import torch
torch.cuda.empty_cache()
```

### Issue: EasyOCR Not Working

```bash
pip uninstall easyocr
pip install easyocr --no-deps
pip install torch torchvision
```

### Issue: Streamlit Port Conflict

```bash
streamlit run app/streamlit_app.py --server.port 8502
```

### Issue: Database Locked

```python
# Close all connections
import sqlite3
conn = sqlite3.connect('database/vehicle_registry.db')
conn.close()
```

---

## 📈 Performance Optimization

### For Faster Training

1. **Use Mixed Precision:**
```python
# Already enabled in VLM trainer
torch_dtype=torch.float16
```

2. **Reduce Image Size:**
```python
# In YOLO training
imgsz=416  # Instead of 640
```

3. **Use Smaller Model:**
```python
# Use YOLOv8n instead of YOLOv8m
model = YOLO('yolov8n.pt')
```

### For Better Accuracy

1. **Increase Epochs:**
```python
epochs=100  # Instead of 50
```

2. **Use Data Augmentation:**
```python
# Already implemented in augmentation.py
```

3. **Ensemble Methods:**
```python
# Combine OCR + VLM predictions
```

---

## 📊 Expected Timeline

| Phase | Duration | GPU Usage |
|-------|----------|-----------|
| Setup | 15 min | 0% |
| YOLO Training | 2-4 hours | 60-70% |
| VLM Fine-tuning | 4-6 hours | 90-95% |
| Evaluation | 30 min | 50-60% |
| Demo | Instant | 20-30% |

**Total:** ~7-11 hours (mostly automated)

---

## ✅ Verification Checklist

- [ ] GPU detected and CUDA available
- [ ] All dependencies installed
- [ ] Datasets unified successfully
- [ ] Database generated (1000 records)
- [ ] YOLO model trained (mAP > 0.80)
- [ ] VLM fine-tuned (loss < 1.5)
- [ ] Evaluation completed
- [ ] Demo app launches successfully
- [ ] Can detect plates from images
- [ ] Can query database
- [ ] Verification engine works

---

## 🎓 Research Outputs

After completing all steps, you will have:

1. **Trained Models:**
   - YOLO detector weights
   - Fine-tuned VLM checkpoint

2. **Evaluation Results:**
   - Comparison plots
   - Metrics JSON
   - Robustness analysis

3. **Demo Application:**
   - Interactive web interface
   - Real-time processing

4. **Database:**
   - 1000 synthetic records
   - Query interface

5. **Documentation:**
   - Complete README
   - Code comments
   - Configuration files

---

## 📝 Next Steps for Publication

1. **Write Paper:**
   - Introduction
   - Related Work
   - Methodology
   - Experiments
   - Results
   - Conclusion

2. **Prepare Figures:**
   - System architecture
   - Comparison plots
   - Qualitative results
   - Failure cases

3. **Create Presentation:**
   - Slides
   - Demo video
   - Poster

4. **Code Release:**
   - GitHub repository
   - Documentation
   - Pre-trained models

---

## 🆘 Support

If you encounter issues:

1. Check this guide
2. Review error messages
3. Check GPU memory: `nvidia-smi`
4. Verify file paths
5. Check Python version
6. Review logs in `outputs/`

---

**Good luck with your research! 🚀**
