import sys
sys.path.append('.')
from vlm.trainer import VLMTrainer

if __name__ == '__main__':
    print("Training VLM with improved approach (20 epochs, lower LR)...")
    trainer = VLMTrainer()
    trainer.train('unified_dataset.json', epochs=20, batch_size=2, lr=5e-5)
    print("Training complete!")
