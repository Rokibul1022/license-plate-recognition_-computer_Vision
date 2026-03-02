"""
VLM Fine-tuning Script with QLoRA
Optimized for RTX 3060 12GB
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm.trainer import VLMTrainer
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Initialize trainer with QLoRA
print("\nInitializing PaliGemma 3B with 4-bit quantization...")
trainer = VLMTrainer("google/paligemma-3b-pt-224")

# Train
print("\nStarting fine-tuning with QLoRA...")
print("Configuration:")
print("- LoRA rank: 16")
print("- LoRA alpha: 32")
print("- Batch size: 2")
print("- Learning rate: 2e-4")
print("- Epochs: 5")

trainer.train(
    train_file="unified_dataset.json",
    epochs=5,
    batch_size=2,
    lr=2e-4
)

print("\nFine-tuning complete!")
print("Model saved to: outputs/vlm/finetuned_model")
