import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
from typing import Dict, List, Tuple

class Trainer:
    """
    Trainer class for EEGNet.
    
    Handles training, validation, history tracking, and early stopping.
    """
    def __init__(self, model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                 val_loader: torch.utils.data.DataLoader, config: Dict, device: str = 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        
        # Learning Rate Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=config.get('patience', 20) // 2
        )
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for inputs, labels in pbar:
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
            
            pbar.set_postfix({'loss': running_loss / total, 'acc': 100. * correct / total})
            
        return running_loss / total, 100. * correct / total

    def validate(self) -> Tuple[float, float]:
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

    def train(self, epochs: int, checkpoint_path: str = 'checkpoints/best_model.pth') -> Dict:
        """
        Runs the training loop for a specified number of epochs.
        """
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Step the scheduler
            self.scheduler.step(val_acc)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
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
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        
        return self.history
