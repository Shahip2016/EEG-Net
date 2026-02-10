import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.
    Paper: Lawhern et al. 2018 (https://arxiv.org/abs/1611.08024)
    """
    def __init__(self, chans=22, samples=1125, dropout_rate=0.5, kern_length=64, f1=8, d=2, f2=16, nb_classes=4):
        super(EEGNet, self).__init__()
        self.chans = chans
        self.samples = samples

        # Block 1 - Temporal Convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(f1)
        )

        # Block 2 - Depthwise Convolution (Spatial Filtering)
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, f1 * d, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        # Block 3 - Separable Convolution
        self.block3 = nn.Sequential(
            # Depthwise
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            # Pointwise
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )

        # Classification block
        # Calculate flattening size dynamically
        self._calculate_flatten_size()
        
        self.classifier = nn.Linear(self.flatten_size, nb_classes)

    def _calculate_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.chans, self.samples)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            self.flatten_size = x.view(1, -1).size(1)

    def forward(self, x):
        # Input shape: (Batch, Channels, Samples) -> Add depth dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Smoke test
    model = EEGNet(chans=22, samples=1125)
    dummy_input = torch.randn(1, 1, 22, 1125)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 4)
    print("EEGNet architecture implemented successfully.")
