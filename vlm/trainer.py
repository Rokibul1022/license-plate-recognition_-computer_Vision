import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from tqdm import tqdm
import numpy as np

class BDPlateDataset(Dataset):
    def __init__(self, data_file, processor):
        with open(data_file, encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        
        # Simplified prompt
        prompt = "<image>read plate"
        answer = f"{item['plate_text']} {item['vehicle_color']} {item['vehicle_type']}"
        full_text = f"{prompt}\n{answer}"
        
        # Process without truncation to avoid image token mismatch
        inputs = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt"
        )
        
        # Manually pad to fixed length
        max_len = 300
        input_ids = inputs['input_ids']
        if input_ids.shape[1] > max_len:
            input_ids = input_ids[:, :max_len]
        else:
            pad_len = max_len - input_ids.shape[1]
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=self.processor.tokenizer.pad_token_id)
        
        # Token type ids
        token_type_ids = torch.zeros_like(input_ids)
        
        # Labels: mask prompt tokens
        labels = input_ids.clone()
        prompt_ids = self.processor.tokenizer(prompt, add_special_tokens=True)['input_ids']
        labels[0, :len(prompt_ids)] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': input_ids.squeeze(0),
            'token_type_ids': token_type_ids.squeeze(0),
            'labels': labels.squeeze(0)
        }

class VLMTrainer:
    def __init__(self, model_name="google/paligemma-3b-pt-224", load_finetuned=False):
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load base model
        base_model = "google/paligemma-3b-pt-224"
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        self.processor = AutoProcessor.from_pretrained(base_model)
        
        if load_finetuned:
            # Load fine-tuned LoRA adapters
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, model_name)
            self.model.train()  # Enable training mode
            print(f"Loaded fine-tuned model from {model_name}")
        else:
            # Prepare for training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # LoRA config
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def train(self, train_file, epochs=5, batch_size=2, lr=2e-4):
        """Train with QLoRA"""
        dataset = BDPlateDataset(train_file, self.processor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Better optimizer with weight decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        # Learning rate scheduler with warmup
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(dataloader),
            pct_start=0.1
        )
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            avg_loss = total_loss/len(dataloader)
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
            
            # Early stopping if loss plateaus
            if epoch > 3 and avg_loss > 11.0:
                print("Loss not decreasing, stopping early")
                break
        
        self.model.save_pretrained("outputs/vlm/finetuned_model")
        self.processor.save_pretrained("outputs/vlm/finetuned_model")
    
    def predict(self, image_input, prompt="<image>read plate"):
        """Inference - accepts file path or numpy array/PIL Image"""
        # Use type() to avoid ultralytics patches interfering with isinstance
        input_type = type(image_input).__name__
        
        if input_type == 'ndarray':
            image = Image.fromarray(image_input).convert('RGB')
        elif 'Image' in input_type:
            image = image_input.convert('RGB')
        else:
            from PIL import Image as PILImage
            image = PILImage.open(image_input).convert('RGB')
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode only generated tokens
        generated_text = self.processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"DEBUG - Generated: {generated_text}")
        
        return self.parse_output(generated_text)
    
    def parse_output(self, text):
        """Parse model output: <plate> <color> <type>"""
        result = {"plate": "UNKNOWN", "color": "unknown", "type": "unknown"}
        
        text = text.strip()
        if not text or any(x in text.lower() for x in ['dovrebbero', 'ambao', 'скачать']):
            return result
        
        # Expected format: <plate_text> <color> <type>
        parts = text.split()
        if len(parts) >= 3:
            result['type'] = parts[-1]
            result['color'] = parts[-2]
            result['plate'] = ' '.join(parts[:-2])
        elif len(parts) == 2:
            result['color'] = parts[-1]
            result['plate'] = parts[0]
        elif len(parts) == 1:
            result['plate'] = parts[0]
        
        return result
