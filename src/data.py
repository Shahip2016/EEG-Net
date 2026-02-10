import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    """
    Custom Dataset for EEG signals.
    Expects data in shape (N, Channels, Samples) and labels in shape (N,)
    """
    def __init__(self, data, labels, transform=None):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

def get_dataloaders(train_data, train_labels, val_data, val_labels, batch_size=64):
    """
    Creates DataLoaders for training and validation sets.
    """
    train_dataset = EEGDataset(train_data, train_labels)
    val_dataset = EEGDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def generate_synthetic_data(n_samples=100, chans=22, samples=1125, n_classes=4):
    """
    Generate synthetic EEG data for testing purposes.
    """
    data = np.random.randn(n_samples, chans, samples)
    labels = np.random.randint(0, n_classes, n_samples)
    return data, labels

if __name__ == "__main__":
    # Test data loader
    data, labels = generate_synthetic_data()
    dataset = EEGDataset(data, labels)
    loader = DataLoader(dataset, batch_size=10)
    
    for batch_data, batch_labels in loader:
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        break
    print("Data layer implemented successfully.")
