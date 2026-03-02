"""
Configuration File for BD License Plate Recognition System
"""

# Hardware Configuration
GPU_DEVICE = 0  # CUDA device ID
MAX_VRAM_GB = 12  # RTX 3060 VRAM

# YOLO Detection Configuration
YOLO_CONFIG = {
    'model': 'yolov8n.pt',  # Nano model for faster training
    'epochs': 50,
    'imgsz': 640,
    'batch': 16,  # Adjust based on VRAM
    'conf_threshold': 0.25,
    'iou_threshold': 0.45
}

# VLM Configuration
VLM_CONFIG = {
    'model_name': 'google/paligemma-3b-pt-224',
    'quantization': '4bit',  # 4-bit for RTX 3060
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'batch_size': 2,  # Small batch for 12GB VRAM
    'learning_rate': 2e-4,
    'epochs': 5,
    'max_length': 512,
    'gradient_checkpointing': True
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'motion_blur_limit': 7,
    'gaussian_blur_limit': 7,
    'brightness_limit': 0.3,
    'contrast_limit': 0.3,
    'rotation_limit': 20,
    'perspective_scale': (0.05, 0.1),
    'noise_var_limit': (10, 50),
    'compression_quality': (60, 100)
}

# Database Configuration
DATABASE_CONFIG = {
    'db_path': 'database/vehicle_registry.db',
    'num_synthetic_records': 1000
}

# Video Processing Configuration
VIDEO_CONFIG = {
    'fps': 5,  # Frame extraction rate
    'frame_output_dir': 'outputs/frames',
    'enable_tracking': False,  # Set True for ByteTrack/DeepSORT
    'duplicate_suppression': True
}

# Evaluation Configuration
EVAL_CONFIG = {
    'test_size': 0.2,
    'metrics': ['plate_accuracy', 'char_accuracy', 'color_accuracy', 'type_accuracy'],
    'robustness_tests': ['blur', 'night', 'angle'],
    'output_dir': 'outputs/evaluation'
}

# Bangladeshi Plate Format
BD_PLATE_CONFIG = {
    'districts': [
        'Dhaka', 'Chattogram', 'Rajshahi', 'Khulna', 'Sylhet', 
        'Barisal', 'Rangpur', 'Mymensingh'
    ],
    'metro_cities': ['Dhaka', 'Chattogram'],
    'class_letters': ['গ', 'ক', 'খ', 'ঘ', 'চ', 'ছ', 'জ', 'ঝ', 'ট', 'ঠ'],
    'format_regex': r'^[A-Za-z\u0980-\u09FF\s]+-[A-Za-z\u0980-\u09FF]\s\d{2}-\d{4}$'
}

# Verification Rules
VERIFICATION_CONFIG = {
    'check_color_match': True,
    'check_type_match': True,
    'check_plate_format': True,
    'check_validity': True,
    'check_travel_time': True,
    'min_travel_time_minutes': 10  # Minimum realistic travel time
}

# Streamlit App Configuration
APP_CONFIG = {
    'title': 'BD License Plate Recognition System',
    'page_icon': '🚗',
    'layout': 'wide',
    'max_upload_size_mb': 200
}

# Paths
PATHS = {
    'data_root': 'data',
    'unified_dataset': 'unified_dataset.json',
    'yolo_weights': 'outputs/detection/plate_detector/weights/best.pt',
    'vlm_checkpoint': 'outputs/vlm/finetuned_model',
    'database': 'database/vehicle_registry.db',
    'outputs': 'outputs'
}

# Training Prompts
PROMPTS = {
    'training': 'Read the Bangladeshi license plate and describe vehicle attributes.',
    'inference': 'Read the Bangladeshi license plate and describe vehicle attributes.',
    'output_format': 'Plate: <text>\nColor: <color>\nType: <type>'
}

# Logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'outputs/system.log'
}
