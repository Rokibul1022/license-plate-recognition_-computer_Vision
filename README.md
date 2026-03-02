# Context-Aware Multitask Vision-Language System for Bangladeshi License Plate Recognition

**Fine-Tuned Multitask Vision-Language Model for Context-Aware Bangladeshi License Plate Recognition and Intelligent Vehicle Information Retrieval from Video**

## 🎓 Academic Research System

**DISCLAIMER:** All personal data used in this research are artificially generated for academic experimentation.

## 📋 System Overview

This is a complete end-to-end intelligent surveillance pipeline that:
- Detects vehicles and license plates from video
- Reads Bangladeshi license plates using Vision-Language Model
- Predicts vehicle attributes (multitask learning)
- Verifies detected data against registry database
- Retrieves contextual vehicle and owner information
- Flags suspicious inconsistencies
- Compares performance against traditional OCR systems

## 🏗️ Architecture

```
Video Input → Frame Extraction → Vehicle Detection (YOLO) → Plate Localization
    ↓
Plate + Vehicle Crop → Vision-Language Model (PaliGemma 3B)
    ↓
Multitask Prediction: Plate Text, Vehicle Color, Vehicle Type
    ↓
Structured Output → Vehicle Registry Database → Context Verification Engine
    ↓
Information Retrieval Interface → Suspicious Activity Detection
```

## 💻 Hardware Requirements

- **GPU:** NVIDIA RTX 3060 (12GB VRAM) or better
- **RAM:** 16GB minimum
- **Storage:** 50GB free space
- **OS:** Windows/Linux with CUDA support

## 📦 Installation

### 1. Clone and Setup Environment

```bash
cd Plate_recogniton_system
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Install Additional Dependencies

```bash
# For Tesseract OCR (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH

# For EasyOCR models
python -c "import easyocr; easyocr.Reader(['bn', 'en'])"
```

## 🚀 Execution Pipeline

### Phase 1: Dataset Unification

```bash
python data_unification.py
```

Unifies three datasets into single format:
- Bangladeshi License Plate Localization & Recognition (~6500)
- Bangladesh Vehicle License Plate Dataset (~2900)
- Bangla License Plate Dataset (~2500)

### Phase 2: Generate Synthetic Database

```bash
python database/generate_db.py
```

Creates SQLite database with 1000 synthetic vehicle records including:
- Owner information
- Vehicle details
- Plate registration
- Location history

### Phase 3: Train YOLO Detector

```bash
python detection/train_yolo.py
```

Trains YOLOv8 for vehicle and plate detection:
- Epochs: 50
- Image size: 640
- Batch size: 16 (adjust for your GPU)
- Output: `outputs/detection/plate_detector/weights/best.pt`

### Phase 4: OCR Baseline (No Training)

OCR baseline uses EasyOCR/Tesseract - ready to use.

### Phase 5: Zero-Shot VLM (No Training)

Uses pre-trained PaliGemma 3B - ready for evaluation.

### Phase 6: Fine-tune VLM with QLoRA

```bash
python vlm/train_vlm.py
```

Fine-tunes PaliGemma 3B with QLoRA:
- 4-bit quantization
- LoRA rank: 16, alpha: 32
- Batch size: 2
- Learning rate: 2e-4
- Epochs: 5
- Memory efficient for RTX 3060

### Phase 7: Evaluation & Comparison

```bash
python evaluation/run_evaluation.py
```

Compares all three methods:
- Plate recognition accuracy
- Character-level accuracy
- Color prediction accuracy
- Type prediction accuracy
- Robustness evaluation

### Phase 8: Launch Demo Application

```bash
streamlit run app/streamlit_app.py
```

Interactive web interface with:
- Image upload and processing
- Video processing
- Database query
- Information retrieval
- Suspicious vehicle detection

## 📊 Expected Results

### Performance Comparison

| Method | Plate Acc | Char Acc | Color Acc | Type Acc |
|--------|-----------|----------|-----------|----------|
| OCR Baseline | ~0.65 | ~0.75 | N/A | N/A |
| Zero-Shot VLM | ~0.70 | ~0.80 | ~0.60 | ~0.55 |
| Fine-tuned VLM | **~0.85** | **~0.90** | **~0.80** | **~0.75** |

### Robustness Evaluation

Fine-tuned VLM shows superior performance under:
- Motion blur
- Low-light/night conditions
- Angled views
- Compression artifacts

## 🗂️ Project Structure

```
Plate_recogniton_system/
├── data/                          # Datasets
├── detection/                     # YOLO detection module
│   ├── detector.py
│   └── train_yolo.py
├── vlm/                          # Vision-Language Model
│   ├── trainer.py
│   └── train_vlm.py
├── baselines/                    # OCR baseline
│   └── ocr_baseline.py
├── database/                     # Synthetic database
│   ├── generate_db.py
│   ├── registry.py
│   └── vehicle_registry.db
├── verification/                 # Context verification
│   └── engine.py
├── video_pipeline/              # Video processing
│   └── processor.py
├── evaluation/                  # Evaluation module
│   ├── evaluator.py
│   └── run_evaluation.py
├── app/                        # Demo application
│   └── streamlit_app.py
├── outputs/                    # Results and models
├── augmentation.py            # Data augmentation
├── data_unification.py        # Dataset unification
├── main_pipeline.py          # Main execution script
└── requirements.txt          # Dependencies
```

## 🔬 Research Contributions

1. **Multitask Vision-Language Model** for joint plate reading and attribute prediction
2. **QLoRA Fine-tuning** optimized for consumer GPU (RTX 3060)
3. **Context-Aware Verification** using synthetic registry database
4. **Comprehensive Evaluation** against traditional OCR methods
5. **Robustness Analysis** under realistic CCTV conditions

## 📝 Data Augmentation

CCTV-realistic augmentations applied:
- Motion blur (simulating moving vehicles)
- Gaussian blur
- Low-light/night simulation
- Rotation (±20°)
- Perspective distortion
- Gaussian noise
- Compression artifacts

## 🔍 Verification Rules

System flags suspicious vehicles when:
- Plate not found in registry
- Color mismatch with registered data
- Vehicle type mismatch
- Invalid plate format
- Registration expired
- Unrealistic travel time between cameras

## 📱 Information Retrieval Options

After plate detection, system provides:
1. **Personal Information** (synthetic)
   - Owner name
   - License number
   - Contact details
   - City

2. **Vehicle Information**
   - Type (Sedan, SUV, etc.)
   - Color
   - Registration year

3. **Plate Details**
   - District
   - Registration year
   - Validity period

4. **Last Known Location**
   - Camera ID
   - Timestamp
   - Location

5. **Movement History**
   - Complete tracking history
   - Timeline visualization

## 🎯 Success Criteria

✅ Fine-tuned VLM > OCR performance  
✅ Robust under distortion conditions  
✅ Multitask prediction works jointly  
✅ Database retrieval operates correctly  
✅ Suspicious detection functions  
✅ End-to-end video demo runs  

## 📄 Citation

If you use this system in your research, please cite:

```bibtex
@misc{bd_lpr_vlm_2024,
  title={Fine-Tuned Multitask Vision-Language Model for Context-Aware Bangladeshi License Plate Recognition},
  author={[Your Name]},
  year={2024},
  note={Academic Research System}
}
```

## ⚠️ Ethical Considerations

- All personal data are **synthetically generated**
- System designed for **academic research only**
- Not intended for real-world surveillance without proper authorization
- Privacy and data protection must be considered in any deployment

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in training scripts
- Use gradient checkpointing (already enabled)
- Close other GPU applications

### EasyOCR Installation Issues
```bash
pip install easyocr --no-deps
pip install torch torchvision
```

### Streamlit Port Already in Use
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

## 📧 Support

For issues and questions:
1. Check troubleshooting section
2. Review error logs in `outputs/`
3. Verify GPU memory availability
4. Ensure all dependencies installed correctly

## 🔄 Future Enhancements

- Real-time video processing
- Multi-camera tracking
- Deep learning-based verification
- Mobile application
- Cloud deployment
- Multi-language support

## 📚 References

- YOLOv8: Ultralytics
- PaliGemma: Google Research
- QLoRA: Dettmers et al., 2023
- EasyOCR: JaidedAI

---

**Built for Academic Research | RTX 3060 Optimized | Production-Ready Architecture**
