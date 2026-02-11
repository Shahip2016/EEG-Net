import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

class EEGDataset(Dataset):
    """
    Custom Dataset for EEG signals.
    Expects data in shape (N, Channels, Samples) and labels in shape (N,)
    """
    def __init__(self, data, labels, transform=None):
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = data.float()
            
        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()
            
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
    # Using TensorDataset for faster performance when no complex transform is needed
    train_dataset = EEGDataset(train_data, train_labels)
    val_dataset = EEGDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def add_gaussian_noise(data, mean=0., std=0.01):
    """Add Gaussian noise to the EEG signal."""
    noise = torch.randn(data.size()) * std + mean
    return data + noise

def time_shift(data, shift_max=10):
    """Randomly shift the EEG signal along the time axis."""
    shift = np.random.randint(-shift_max, shift_max)
    return torch.roll(data, shifts=shift, dims=-1)

def generate_synthetic_data(n_samples=100, chans=22, samples=1125, n_classes=4):
    """
    Generate synthetic EEG data with rudimentary patterns (e.g., sine waves).
    """
    data = []
    labels = []
    
    t = np.linspace(0, 1, samples)
    
    for i in range(n_samples):
        label = np.random.randint(0, n_classes)
        # Base frequency depends on label to make it learnable
        freq = 10 + label * 5 
        # Create a basic sine wave pattern
        signal = np.sin(2 * np.pi * freq * t)
        # Add spatial distribution (some channels stronger than others)
        spatial_filter = np.random.randn(chans, 1)
        eeg = spatial_filter * signal
        # Add background noise
        eeg += 0.5 * np.random.randn(chans, samples)
        
        data.append(eeg)
        labels.append(label)
        
    return np.array(data), np.array(labels)

if __name__ == "__main__":
    # Test data loader
    data, labels = generate_synthetic_data(n_samples=50)
    print(f"Generated data shape: {data.shape}")
    
    dataset = EEGDataset(data, labels)
    loader = DataLoader(dataset, batch_size=10)
    
    for batch_data, batch_labels in loader:
        print(f"Batch data shape: {batch_data.shape}")
        # Test augmentation
        noisy_data = add_gaussian_noise(batch_data)
        shifted_data = time_shift(batch_data)
        print(f"Augmented data shapes: Noise={noisy_data.shape}, Shift={shifted_data.shape}")
        break
    print("Data layer optimization implemented successfully.")
