import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
from tqdm import tqdm

class AttributeDataset(Dataset):
    def __init__(self, data_file, transform=None):
        with open(data_file, encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Create label mappings
        self.colors = list(set(item['vehicle_color'] for item in self.data))
        self.types = list(set(item['vehicle_type'] for item in self.data))
        
        self.color_to_idx = {c: i for i, c in enumerate(self.colors)}
        self.type_to_idx = {t: i for i, t in enumerate(self.types)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        image = self.transform(image)
        
        color_label = self.color_to_idx[item['vehicle_color']]
        type_label = self.type_to_idx[item['vehicle_type']]
        
        return image, color_label, type_label

class AttributeClassifier(nn.Module):
    def __init__(self, num_colors, num_types):
        super().__init__()
        # Use ResNet18 backbone
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        
        # Remove original FC layer
        self.backbone.fc = nn.Identity()
        
        # Separate heads for color and type
        self.color_head = nn.Linear(num_features, num_colors)
        self.type_head = nn.Linear(num_features, num_types)
        
    def forward(self, x):
        features = self.backbone(x)
        color_logits = self.color_head(features)
        type_logits = self.type_head(features)
        return color_logits, type_logits

class AttributeTrainer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.dataset = None
        
    def train(self, train_file, epochs=10, batch_size=32, lr=1e-3):
        self.dataset = AttributeDataset(train_file)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        self.model = AttributeClassifier(
            num_colors=len(self.dataset.colors),
            num_types=len(self.dataset.types)
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct_color = 0
            correct_type = 0
            total = 0
            
            for images, color_labels, type_labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device)
                color_labels = color_labels.to(self.device)
                type_labels = type_labels.to(self.device)
                
                color_logits, type_logits = self.model(images)
                
                loss_color = criterion(color_logits, color_labels)
                loss_type = criterion(type_logits, type_labels)
                loss = loss_color + loss_type
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, color_pred = torch.max(color_logits, 1)
                _, type_pred = torch.max(type_logits, 1)
                correct_color += (color_pred == color_labels).sum().item()
                correct_type += (type_pred == type_labels).sum().item()
                total += color_labels.size(0)
            
            avg_loss = total_loss / len(dataloader)
            color_acc = 100 * correct_color / total
            type_acc = 100 * correct_type / total
            
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Color Acc: {color_acc:.2f}%, Type Acc: {type_acc:.2f}%")
        
        # Save model
        torch.save({
            'model_state': self.model.state_dict(),
            'colors': self.dataset.colors,
            'types': self.dataset.types
        }, "outputs/trocr/attribute_classifier.pth")
    
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.colors = checkpoint['colors']
        self.types = checkpoint['types']
        
        self.model = AttributeClassifier(
            num_colors=len(self.colors),
            num_types=len(self.types)
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
    
    def predict(self, image):
        if self.model is None:
            raise ValueError("Model not loaded")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Handle different input types
        if type(image).__name__ == 'ndarray':
            image = Image.fromarray(image).convert('RGB')
        elif not type(image).__name__ == 'Image':
            image = Image.open(image).convert('RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            color_logits, type_logits = self.model(image_tensor)
            color_idx = torch.argmax(color_logits, dim=1).item()
            type_idx = torch.argmax(type_logits, dim=1).item()
        
        return {
            'color': self.colors[color_idx],
            'type': self.types[type_idx]
        }
