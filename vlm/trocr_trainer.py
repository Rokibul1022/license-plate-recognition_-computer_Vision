import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from tqdm import tqdm

class PlateTextDataset(Dataset):
    def __init__(self, data_file, processor):
        with open(data_file, encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        
        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Tokenize text
        labels = self.processor.tokenizer(
            item['plate_text'],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        # Replace padding token id with -100 for loss calculation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }

class TrOCRTrainer:
    def __init__(self):
        # Use base TrOCR model
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Set special tokens
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        # Move to GPU
        self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, train_file, epochs=10, batch_size=8, lr=5e-5):
        dataset = PlateTextDataset(train_file, self.processor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                pixel_values = batch['pixel_values'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
            
            if avg_loss < 0.5:
                print("Loss converged, stopping early")
                break
        
        # Save model
        self.model.save_pretrained("outputs/trocr/plate_reader")
        self.processor.save_pretrained("outputs/trocr/plate_reader")
    
    def predict(self, image):
        self.model.eval()
        
        # Handle different input types
        if type(image).__name__ == 'ndarray':
            image = Image.fromarray(image).convert('RGB')
        elif not type(image).__name__ == 'Image':
            image = Image.open(image).convert('RGB')
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=64)
        
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text
