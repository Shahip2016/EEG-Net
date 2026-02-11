import torch
import torch.nn as nn
from typing import Tuple

class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.
    
    This implementation follows the architecture described in:
    Lawhern et al. 2018 (https://arxiv.org/abs/1611.08024)
    
    Attributes:
        chans (int): Number of EEG channels.
        samples (int): Number of time points in the EEG signal.
        nb_classes (int): Number of output classes.
    """
    def __init__(
        self, 
        chans: int = 22, 
        samples: int = 1125, 
        dropout_rate: float = 0.5, 
        kern_length: int = 64, 
        f1: int = 8, 
        d: int = 2, 
        f2: int = 16, 
        nb_classes: int = 4
    ):
        super(EEGNet, self).__init__()
        self.chans = chans
        self.samples = samples
        self.nb_classes = nb_classes

        # Block 1 - Temporal Convolution
        # Learns frequency-specific filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(f1)
        )

        # Block 2 - Depthwise Convolution (Spatial Filtering)
        # Learns frequency-specific spatial filters
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, f1 * d, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        # Block 3 - Separable Convolution
        # Learning temporal summaries and then combining them
        self.block3 = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            # Pointwise convolution
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )

        # Classification block
        self.flatten_size = self._get_flatten_size()
        self.classifier = nn.Linear(self.flatten_size, nb_classes)

    def _get_flatten_size(self) -> int:
        """Calculate the flattening size dynamically based on input dimensions."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.chans, self.samples)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of EEGNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Samples) 
                              or (Batch, 1, Channels, Samples).
        Returns:
            torch.Tensor: Logits for each class.
        """
        # Ensure input has depth dimension (Batch, Depth, Chans, Samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Smoke test for model architecture
    chans, samples = 22, 1125
    model = EEGNet(chans=chans, samples=samples)
    
    # Test batch input
    dummy_input = torch.randn(2, 1, chans, samples)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (2, 4), f"Expected (2, 4), got {output.shape}"
    print("EEGNet optimization verified successfully.")
