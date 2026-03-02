# Bangladeshi License Plate Recognition System - Technical Report

## Executive Summary

This project developed an end-to-end intelligent surveillance system for Bangladeshi license plate recognition using deep learning. The system integrates YOLOv8 for detection, multiple recognition approaches (OCR, TrOCR, PaliGemma), and a context-aware verification engine with synthetic database.

**Key Achievement:** Successfully demonstrated that task-specific models and data quality are more critical than model size for specialized OCR tasks.

---

## 1. System Architecture

### 1.1 Pipeline Overview
```
Video/Image Input → YOLOv8 Detection → Plate Localization → Recognition (OCR/TrOCR) 
    ↓
Attribute Classification → Database Verification → Information Retrieval → Alert System
```

### 1.2 Components
- **Detection Module**: YOLOv8 (50 epochs, mAP >0.85)
- **Recognition Module**: EasyOCR, TrOCR, PaliGemma (comparative study)
- **Attribute Classifier**: ResNet18-based multitask CNN
- **Database**: SQLite with 1,000 synthetic vehicle records
- **Verification Engine**: Rule-based context verification
- **Interface**: Streamlit web application

---

## 2. Dataset

### 2.1 Composition
- **Total Images**: 1,694 vehicle images
- **Sources**: 3 public Bangladeshi license plate datasets
- **Format**: Unified JSON with bbox, plate_text, color, type
- **Languages**: Bangla and English text
- **Vehicle Types**: Sedan, SUV, Truck, Bus, Motorcycle, CNG (6 classes)
- **Colors**: Red, Blue, Green, White, Black (5 classes)

### 2.2 Data Quality Issues
- **Low resolution**: Many plate crops <100x50 pixels
- **Motion blur**: Significant blur in video frames
- **Label accuracy**: Auto-generated labels using EasyOCR (60-70% accuracy)
- **Inconsistent lighting**: Day/night variations
- **Occlusions**: Partial plate visibility in some images

---

## 3. Model Training & Results

### 3.1 YOLOv8 Detection
**Configuration:**
- Model: YOLOv8n (nano)
- Epochs: 50
- Image size: 640x640
- Batch size: 16
- Classes: vehicle, license_plate

**Results:**
- ✅ Training Loss: 0.8 → 0.3
- ✅ mAP@0.5: 0.87
- ✅ Inference time: 42ms per image
- ✅ Status: **SUCCESSFUL**

### 3.2 PaliGemma 3B Fine-Tuning (FAILED)
**Configuration:**
- Model: google/paligemma-3b-pt-224
- Method: QLoRA (4-bit quantization)
- LoRA: rank=16, alpha=32
- Epochs: 15
- Batch size: 2
- Learning rate: 5e-5

**Results:**
- ❌ Training Loss: 11.7 → 11.5 (no convergence)
- ❌ Plate Recognition: Hallucinated outputs
- ❌ Root Cause: Model too large for dataset size, task mismatch
- ❌ Status: **FAILED**

**Key Learning:** General-purpose VLMs require 50K+ samples for specialized tasks.

### 3.3 TrOCR Fine-Tuning (PARTIAL SUCCESS)
**Configuration:**
- Model: microsoft/trocr-base-handwritten
- Epochs: 10
- Batch size: 8
- Learning rate: 5e-5
- Optimizer: AdamW

**Results:**
- ⚠️ Training Loss: 5.0 → 1.5 (converged)
- ⚠️ Plate Recognition: Inaccurate due to label quality
- ⚠️ Root Cause: Trained on incorrect auto-generated labels
- ⚠️ Status: **TECHNICALLY SUCCESSFUL, PRACTICALLY FAILED**

**Key Learning:** Model convergence ≠ practical accuracy. Clean labels essential.

### 3.4 Attribute Classifier (PARTIAL SUCCESS)
**Configuration:**
- Backbone: ResNet18 (ImageNet pretrained)
- Heads: Separate for color and type
- Epochs: 10
- Batch size: 32
- Learning rate: 1e-3

**Results:**
- ⚠️ Training Accuracy: Color 85%, Type 82%
- ⚠️ Test Accuracy: Lower due to training on plate crops instead of full vehicles
- ⚠️ Root Cause: Input mismatch during training
- ⚠️ Status: **NEEDS RETRAINING**

### 3.5 EasyOCR Baseline (BEST PERFORMER)
**Configuration:**
- Languages: Bengali, English
- No training required

**Results:**
- ✅ Plate Recognition: 60-70% accuracy
- ✅ Bangla Support: Native
- ✅ Inference: 200ms per plate
- ✅ Status: **MOST RELIABLE**

---

## 4. Comparative Analysis

| Method | Params | Training Time | Loss | Plate Acc | Status |
|--------|--------|---------------|------|-----------|--------|
| EasyOCR | - | None | - | 60-70% | ✅ Works |
| TrOCR | 300M | 2 hours | 1.5 | <30% | ⚠️ Bad labels |
| PaliGemma | 3B | 5 hours | 11.5 | 0% | ❌ Failed |
| ResNet18 | 11M | 30 mins | 0.8 | 70%* | ⚠️ Needs fix |

*Accuracy on training set; test accuracy lower

---

## 5. Database & Verification

### 5.1 Synthetic Database
- **Records**: 1,000 vehicles
- **Tables**: owners, vehicles, plates, locations
- **Fields**: Name, license, contact, vehicle details, timestamps
- **Purpose**: Context-aware verification without privacy concerns

### 5.2 Verification Rules
System flags suspicious vehicles when:
1. Plate not found in registry
2. Color mismatch (predicted vs registered)
3. Type mismatch (predicted vs registered)
4. Invalid plate format
5. Registration expired
6. Unrealistic travel time between cameras

---

## 6. Challenges & Limitations

### 6.1 Technical Challenges
1. **Data Quality**: Low-resolution, blurry images
2. **Label Accuracy**: Auto-generated labels unreliable
3. **Model Selection**: PaliGemma too complex for task
4. **Resource Constraints**: RTX 3060 12GB VRAM limiting
5. **Bangla OCR**: Limited pre-trained models

### 6.2 Limitations
1. **Recognition Accuracy**: 60-70% (below production threshold of 95%)
2. **Real-time Performance**: 250ms per frame (4 FPS)
3. **Robustness**: Poor performance on night/blur images
4. **Generalization**: Trained on specific dataset characteristics
5. **Database**: Synthetic data, not real vehicle registry

---

## 7. Key Findings & Contributions

### 7.1 Research Findings
1. **Task-specific models outperform general VLMs** on specialized tasks with limited data
2. **Data quality > Model size** for practical accuracy
3. **Label correctness critical** for supervised learning success
4. **TrOCR (300M) > PaliGemma (3B)** for OCR tasks
5. **Traditional OCR competitive** with deep learning on low-resource scenarios

### 7.2 Technical Contributions
1. Complete end-to-end LPR system for Bangladeshi plates
2. Unified dataset from 3 public sources (1,694 images)
3. Comparative study: OCR vs TrOCR vs PaliGemma
4. Context-aware verification engine
5. Synthetic database generation methodology
6. RTX 3060 optimization strategies

### 7.3 Practical Contributions
1. Streamlit demo application
2. Video processing pipeline
3. Information retrieval interface
4. Suspicious vehicle detection
5. Database query system

---

## 8. Future Work

### 8.1 Immediate Improvements
1. **Manual labeling**: 500-1000 plates for clean training data
2. **Data augmentation**: Synthetic plate generation
3. **Ensemble methods**: Combine OCR + TrOCR predictions
4. **Post-processing**: Regex-based plate format validation
5. **Attribute classifier fix**: Train on full vehicle images

### 8.2 Long-term Enhancements
1. **Real-time optimization**: TensorRT/ONNX conversion
2. **Multi-camera tracking**: Vehicle re-identification
3. **Deep learning verification**: Replace rule-based system
4. **Mobile deployment**: Edge device optimization
5. **Real database integration**: Government vehicle registry API
6. **Multi-language support**: English-only plate recognition
7. **Attention mechanisms**: Focus on plate region for attributes

---

## 9. Deployment Considerations

### 9.1 Hardware Requirements
- **Minimum**: NVIDIA GTX 1660 (6GB VRAM)
- **Recommended**: NVIDIA RTX 3060 (12GB VRAM)
- **CPU**: Intel i5 or AMD Ryzen 5
- **RAM**: 16GB
- **Storage**: 50GB SSD

### 9.2 Software Stack
- Python 3.12
- PyTorch 2.7.1
- CUDA 11.8
- Transformers 4.x
- Ultralytics YOLOv8
- Streamlit 1.x
- SQLite 3

### 9.3 Performance Metrics
- **Detection**: 42ms per image
- **Recognition**: 200ms per plate
- **Total Pipeline**: 250ms per frame
- **Throughput**: 4 FPS
- **Memory**: 10GB VRAM peak

---

## 10. Conclusion

This project successfully developed a complete Bangladeshi license plate recognition system with detection, recognition, and verification components. While YOLOv8 detection achieved production-ready accuracy (87% mAP), recognition accuracy remained limited (60-70%) due to dataset quality issues.

**Key Takeaway:** The project demonstrates that successful deep learning systems require not just sophisticated models, but high-quality labeled data. PaliGemma 3B's failure and TrOCR's limited success, despite proper training procedures, highlight that data quality is the primary bottleneck in specialized OCR tasks.

**Academic Value:** This work provides empirical evidence for model selection in low-resource scenarios and documents real-world challenges in deploying vision-language models for specialized tasks.

---

## 11. References

1. Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
2. PaliGemma: Google Research, 2024
3. TrOCR: Microsoft Research, 2021
4. EasyOCR: JaidedAI
5. Bangladeshi License Plate Datasets: Kaggle/Roboflow

---

## 12. Acknowledgments

- Dataset providers: Kaggle community
- Pre-trained models: Google, Microsoft, Ultralytics
- Hardware: NVIDIA RTX 3060
- Framework: PyTorch, Transformers, Streamlit

---

**Project Status:** Proof-of-concept complete, production deployment requires data quality improvements.

**Recommended Next Steps:** Manual labeling of 500 plates, retrain TrOCR, deploy with ensemble OCR+TrOCR.
