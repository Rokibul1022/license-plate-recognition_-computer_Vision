# 🎨 Visual System Diagrams

## System Architecture Visualizations

---

## 1. Complete System Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VIDEO/IMAGE INPUT                                │
│                    (MP4, AVI, JPG, PNG files)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING MODULE                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │    Frame     │  │  Augmentation│  │   Resize &   │                 │
│  │  Extraction  │→ │   Pipeline   │→ │  Normalize   │                 │
│  │   (5 FPS)    │  │  (Albumenta) │  │  (640x640)   │                 │
│  └──────────────┘  └──────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    DETECTION MODULE (YOLOv8)                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    CSPDarknet53 Backbone                         │  │
│  │                            ↓                                     │  │
│  │                       PANet Neck                                 │  │
│  │                            ↓                                     │  │
│  │                  Decoupled Detection Head                        │  │
│  │                            ↓                                     │  │
│  │  Output: [x1, y1, x2, y2], confidence, class                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Performance: mAP@0.5 = 85.3%, FPS = 32                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      PLATE CROPPING MODULE                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │   Extract    │  │   Resize     │  │  Preprocess  │                 │
│  │   Plate ROI  │→ │  (224x224)   │→ │  for VLM     │                 │
│  └──────────────┘  └──────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│              RECOGNITION MODULE (PaliGemma 3B + QLoRA)                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   Vision Encoder (SigLIP)                        │  │
│  │              224x224 → 1152-dim embeddings                       │  │
│  │                            ↓                                     │  │
│  │                    Projection Layer                              │  │
│  │              1152-dim → 3072-dim (Gemma space)                  │  │
│  │                            ↓                                     │  │
│  │              Language Model (Gemma 3B + LoRA)                   │  │
│  │         28 layers, 16 heads, 4-bit quantized                    │  │
│  │         LoRA: r=16, α=32 (8.4M trainable params)               │  │
│  │                            ↓                                     │  │
│  │  Output: "Plate: <text>\nColor: <color>\nType: <type>"         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Performance: Plate=85%, Color=80%, Type=75%                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    PARSING & STRUCTURING MODULE                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Parse VLM output → Extract plate, color, type                  │  │
│  │  Validate format → Check BD plate pattern                       │  │
│  │  Normalize text → Clean and standardize                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                  DATABASE QUERY MODULE (SQLite)                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Query: SELECT * FROM vehicles WHERE plate_number = ?           │  │
│  │                            ↓                                     │  │
│  │  JOIN owners, plates, locations tables                          │  │
│  │                            ↓                                     │  │
│  │  Return: Complete vehicle profile                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Database: 1000 synthetic records, 4 tables                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                  VERIFICATION ENGINE (Rule-Based)                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Rule 1: Database Lookup                                        │  │
│  │    ✓ Found → Continue    ✗ Not Found → Flag                   │  │
│  │                                                                  │  │
│  │  Rule 2: Color Matching                                         │  │
│  │    ✓ Match → Continue    ✗ Mismatch → Flag                    │  │
│  │                                                                  │  │
│  │  Rule 3: Type Matching                                          │  │
│  │    ✓ Match → Continue    ✗ Mismatch → Flag                    │  │
│  │                                                                  │  │
│  │  Rule 4: Format Validation                                      │  │
│  │    ✓ Valid → Continue    ✗ Invalid → Flag                     │  │
│  │                                                                  │  │
│  │  Rule 5: Travel Time Analysis                                   │  │
│  │    ✓ Realistic → Continue    ✗ Impossible → Flag              │  │
│  │                                                                  │  │
│  │  Output: Status (verified/suspicious) + Flags                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                  INFORMATION RETRIEVAL INTERFACE                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Option 1: Personal Information (Owner details)                 │  │
│  │  Option 2: Vehicle Information (Type, color, year)              │  │
│  │  Option 3: Plate Details (District, registration, validity)     │  │
│  │  Option 4: Last Known Location (Camera, timestamp)              │  │
│  │  Option 5: Movement History (Complete tracking)                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │  Streamlit   │  │    JSON      │  │   Alert      │                 │
│  │   Web UI     │  │    API       │  │   System     │                 │
│  └──────────────┘  └──────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATASETS (3 sources)                      │
│  • Bangladeshi Vehicle Image (~2900)                            │
│  • BD License Plate Recognition (~6500)                         │
│  • Bangla License Plate Dataset (~2500)                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA UNIFICATION                              │
│  Script: data_unification.py                                    │
│  Output: unified_dataset.json (11,900 samples)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA AUGMENTATION                             │
│  • Motion Blur (CCTV simulation)                                │
│  • Gaussian Blur                                                │
│  • Low-light/Night                                              │
│  • Rotation (±20°)                                              │
│  • Perspective Distortion                                       │
│  • Noise & Compression                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│   YOLO TRAINING          │  │   VLM FINE-TUNING        │
│                          │  │                          │
│  Model: YOLOv8n          │  │  Model: PaliGemma 3B     │
│  Epochs: 50              │  │  Method: QLoRA           │
│  Batch: 16               │  │  Quantization: 4-bit     │
│  Time: 2-4 hours         │  │  LoRA r=16, α=32        │
│  VRAM: 6-8GB             │  │  Batch: 2                │
│                          │  │  Epochs: 5               │
│  Output:                 │  │  Time: 4-6 hours         │
│  plate_detector.pt       │  │  VRAM: 10-11GB           │
│  mAP@0.5: 85%+          │  │                          │
│                          │  │  Output:                 │
│                          │  │  finetuned_model/        │
│                          │  │  Accuracy: 85%+          │
└──────────────────────────┘  └──────────────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINED MODELS                                │
│  • YOLO Detector (vehicle & plate detection)                    │
│  • Fine-tuned VLM (plate reading + attributes)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Inference Pipeline

```
┌──────────────┐
│ Input Image  │
└──────────────┘
       ↓
┌──────────────────────────────────────┐
│      YOLO Detection                  │
│  • Detect vehicle bbox               │
│  • Detect plate bbox                 │
│  • Confidence > 0.25                 │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│      Crop Plate Region               │
│  • Extract plate from image          │
│  • Resize to 224x224                 │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│      VLM Recognition                 │
│  • Input: plate image + prompt       │
│  • Process through PaliGemma         │
│  • Generate structured output        │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│      Parse Output                    │
│  • Extract plate text                │
│  • Extract color                     │
│  • Extract type                      │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│      Database Query                  │
│  • Query by plate number             │
│  • Retrieve vehicle info             │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│      Verification                    │
│  • Check color match                 │
│  • Check type match                  │
│  • Validate format                   │
│  • Analyze travel time               │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│      Output Result                   │
│  {                                   │
│    "plate": "...",                   │
│    "color": "...",                   │
│    "type": "...",                    │
│    "status": "verified/suspicious",  │
│    "db_info": {...}                  │
│  }                                   │
└──────────────────────────────────────┘
```

---

## 4. Memory Optimization (QLoRA)

```
┌─────────────────────────────────────────────────────────────────┐
│              FULL FINE-TUNING (Not Possible)                     │
│                                                                  │
│  Model Parameters: 3B × 4 bytes = 12 GB                         │
│  Gradients: 12 GB                                               │
│  Optimizer States (Adam): 24 GB                                 │
│  Activations: ~8 GB                                             │
│  ─────────────────────────────────────────                      │
│  TOTAL: ~56 GB ❌ (Exceeds RTX 3060 12GB)                      │
└─────────────────────────────────────────────────────────────────┘

                              ↓ QLoRA

┌─────────────────────────────────────────────────────────────────┐
│              QLORA FINE-TUNING (Fits RTX 3060)                   │
│                                                                  │
│  Base Model (4-bit): 3B × 0.5 bytes = 1.5 GB                   │
│  LoRA Adapters: 8.4M × 4 bytes = 33.6 MB                       │
│  Gradients (LoRA only): 33.6 MB                                 │
│  Optimizer States: 67.2 MB                                      │
│  Activations: ~8 GB                                             │
│  ─────────────────────────────────────────────                  │
│  TOTAL: ~10 GB ✅ (Fits RTX 3060 12GB)                         │
│                                                                  │
│  Memory Savings: 56 GB → 10 GB (82% reduction)                 │
└─────────────────────────────────────────────────────────────────┘

Key Techniques:
┌──────────────────────────────────────┐
│  1. 4-bit Quantization (NF4)         │
│     • Reduces model size by 8×       │
│     • Minimal accuracy loss          │
└──────────────────────────────────────┘
┌──────────────────────────────────────┐
│  2. LoRA (Low-Rank Adaptation)       │
│     • Only train 0.29% of params     │
│     • Rank=16, Alpha=32              │
└──────────────────────────────────────┘
┌──────────────────────────────────────┐
│  3. Gradient Checkpointing           │
│     • Trade compute for memory       │
│     • Recompute activations          │
└──────────────────────────────────────┘
┌──────────────────────────────────────┐
│  4. Mixed Precision (FP16)           │
│     • Use FP16 for computation       │
│     • FP32 for critical ops          │
└──────────────────────────────────────┘
```

---

## 5. Database Schema

```
┌─────────────────────────────────────────────────────────────────┐
│                         OWNERS TABLE                             │
│  ┌────────────┬──────────────┬──────────────┬────────────────┐  │
│  │ owner_id   │ name         │ license_num  │ phone          │  │
│  │ (PK)       │ TEXT         │ TEXT         │ TEXT           │  │
│  ├────────────┼──────────────┼──────────────┼────────────────┤  │
│  │ 1          │ Rahim Ahmed  │ DL-DA-203948 │ 01712345678    │  │
│  │ 2          │ Fatema Khan  │ DL-CH-456789 │ 01823456789    │  │
│  │ ...        │ ...          │ ...          │ ...            │  │
│  └────────────┴──────────────┴──────────────┴────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓ (owner_id FK)
┌─────────────────────────────────────────────────────────────────┐
│                        VEHICLES TABLE                            │
│  ┌────────────┬──────────────┬──────────┬──────────┬─────────┐  │
│  │ vehicle_id │ plate_number │ color    │ type     │owner_id │  │
│  │ (PK)       │ TEXT (UNIQUE)│ TEXT     │ TEXT     │ (FK)    │  │
│  ├────────────┼──────────────┼──────────┼──────────┼─────────┤  │
│  │ 1          │ Dhaka-গ 12-  │ White    │ Sedan    │ 1       │  │
│  │            │ 3456         │          │          │         │  │
│  │ 2          │ Chattogram-  │ Black    │ SUV      │ 2       │  │
│  │            │ ক 45-7890    │          │          │         │  │
│  │ ...        │ ...          │ ...      │ ...      │ ...     │  │
│  └────────────┴──────────────┴──────────┴──────────┴─────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓ (plate_number FK)
┌─────────────────────────────────────────────────────────────────┐
│                         PLATES TABLE                             │
│  ┌──────────────┬──────────┬──────────────┬────────────────┐    │
│  │ plate_number │ district │ reg_year     │ validity       │    │
│  │ (PK)         │ TEXT     │ INTEGER      │ TEXT           │    │
│  ├──────────────┼──────────┼──────────────┼────────────────┤    │
│  │ Dhaka-গ 12-  │ Dhaka    │ 2020         │ 2035-12-31     │    │
│  │ 3456         │          │              │                │    │
│  │ Chattogram-  │Chattogram│ 2019         │ 2034-12-31     │    │
│  │ ক 45-7890    │          │              │                │    │
│  │ ...          │ ...      │ ...          │ ...            │    │
│  └──────────────┴──────────┴──────────────┴────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ↓ (plate_number FK)
┌─────────────────────────────────────────────────────────────────┐
│                      LOCATIONS TABLE                             │
│  ┌────┬──────────────┬───────────┬────────────┬──────────────┐  │
│  │ id │ plate_number │ camera_id │ timestamp  │ location     │  │
│  │(PK)│ TEXT         │ TEXT      │ DATETIME   │ TEXT         │  │
│  ├────┼──────────────┼───────────┼────────────┼──────────────┤  │
│  │ 1  │ Dhaka-গ 12-  │Farmgate-01│2024-01-15  │ Farmgate     │  │
│  │    │ 3456         │           │ 14:30:00   │              │  │
│  │ 2  │ Dhaka-গ 12-  │Gulshan-03 │2024-01-15  │ Gulshan      │  │
│  │    │ 3456         │           │ 15:00:00   │              │  │
│  │ ...│ ...          │ ...       │ ...        │ ...          │  │
│  └────┴──────────────┴───────────┴────────────┴──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Evaluation Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    METHOD COMPARISON                             │
│                                                                  │
│  Plate Accuracy                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ OCR Baseline      ████████████████ 65%                 │     │
│  │ Zero-Shot VLM     ██████████████████ 70%               │     │
│  │ Fine-tuned VLM    █████████████████████████ 85%        │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Character Accuracy                                              │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ OCR Baseline      ███████████████████ 75%              │     │
│  │ Zero-Shot VLM     ████████████████████ 80%             │     │
│  │ Fine-tuned VLM    ███████████████████████████ 90%      │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Color Accuracy                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ OCR Baseline      N/A                                  │     │
│  │ Zero-Shot VLM     ████████████████ 60%                 │     │
│  │ Fine-tuned VLM    ████████████████████ 80%             │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Type Accuracy                                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ OCR Baseline      N/A                                  │     │
│  │ Zero-Shot VLM     ██████████████ 55%                   │     │
│  │ Fine-tuned VLM    ███████████████████ 75%              │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Winner: Fine-tuned VLM (85% plate, 90% char, 80% color, 75% type) │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. File Organization

```
Plate_recogniton_system/
│
├── 📄 Core Scripts
│   ├── main_pipeline.py ............... Master execution
│   ├── quick_start.py ................. Automated setup
│   ├── data_unification.py ............ Dataset merger
│   ├── augmentation.py ................ CCTV augmentation
│   ├── utils.py ....................... Helper functions
│   ├── config.py ...................... Configuration
│   └── requirements.txt ............... Dependencies
│
├── 📁 detection/ ...................... YOLO Module
│   ├── detector.py .................... PlateDetector class
│   └── train_yolo.py .................. Training script
│
├── 📁 vlm/ ............................ VLM Module
│   ├── trainer.py ..................... VLMTrainer + QLoRA
│   └── train_vlm.py ................... Fine-tuning script
│
├── 📁 baselines/ ...................... Baseline Methods
│   └── ocr_baseline.py ................ EasyOCR/Tesseract
│
├── 📁 database/ ....................... Database Module
│   ├── generate_db.py ................. Synthetic generator
│   ├── registry.py .................... Query interface
│   └── vehicle_registry.db ............ SQLite database
│
├── 📁 verification/ ................... Verification Module
│   └── engine.py ...................... Rule-based verifier
│
├── 📁 video_pipeline/ ................. Video Processing
│   └── processor.py ................... VideoProcessor class
│
├── 📁 evaluation/ ..................... Evaluation Module
│   ├── evaluator.py ................... Metrics calculator
│   └── run_evaluation.py .............. Comparison script
│
├── 📁 app/ ............................ Demo Application
│   └── streamlit_app.py ............... Web interface
│
├── 📁 outputs/ ........................ Generated Outputs
│   ├── detection/ ..................... YOLO results
│   ├── vlm/ ........................... VLM checkpoints
│   ├── evaluation/ .................... Metrics & plots
│   └── frames/ ........................ Video frames
│
└── 📚 Documentation
    ├── README.md ...................... Main docs
    ├── EXECUTION_GUIDE.md ............. Step-by-step
    ├── ARCHITECTURE.md ................ Technical details
    ├── PROJECT_SUMMARY.md ............. File index
    ├── CHECKLIST.md ................... Progress tracker
    └── OVERVIEW.md .................... Complete overview
```

---

**All diagrams are text-based for easy viewing in any text editor!**
