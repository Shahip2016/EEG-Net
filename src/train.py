import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        
        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / total, 100. * correct / total

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return running_loss / total, 100. * correct / total

    def train(self, epochs, checkpoint_path='checkpoints/best_model.pth'):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
            
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping and checkpointing
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), checkpoint_path)
                self.patience_counter = 0
                print(f"--> Saved best model with Val Acc: {val_acc:.2f}%")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.get('patience', 20):
                print("Early stopping triggered.")
                break
        
        return self.best_val_acc
