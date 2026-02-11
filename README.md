# EEGNet-Startup

A professional, startup-grade implementation of the **EEGNet** architecture (Lawhern et al. 2018) for EEG signal classification.

## 🚀 Features

- **Core Architecture**: Refactored PyTorch implementation of EEGNet with type hints and optimized conv blocks.
- **Data Augmentation**: Built-in Gaussian noise and time-shifting for robust model training.
- **Reproducibility**: Global seed control for consistent experiments via CLI.
- **Professional CLI**: Comprehensive command-line interface with configuration overrides.
- **Advanced Visualization**: High-quality plots for training history (Loss/Acc) and normalized confusion matrices.
- **LR Scheduler**: Automatic learning rate reduction on plateau for better convergence.

## 🛠️ System Requirements

> [!IMPORTANT]
> This project requires **64-bit Python** (3.8+) due to PyTorch compatibility on Windows.

## 📦 Installation

```bash
git clone git@github.com:Shahip2016/EEG-Net.git
cd EEG-Net
py -3.14-64 -m pip install -r requirements.txt
```

## 📖 Usage

### Training
To train the model with the default configuration and a specific seed:
```bash
python main.py train --config config/base_config.yaml --seed 42 --output outputs/
```

### Evaluation
To evaluate a trained model and generate a detailed performance report:
```bash
python main.py evaluate --model_path outputs/best_model.pth --config config/base_config.yaml
```

## 📊 Architecture Details
EEGNet utilizes a specialized CNN structure:
1. **Temporal Convolution**: Learns frequency-specific features.
2. **Depthwise Convolution**: Captures spatial filters across channels.
3. **Separable Convolution**: Combines temporal summaries while maintaining efficiency.

## 📚 References
- Lawhern, V. J., et al. (2018). "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces." *Journal of Neural Engineering*. [Link](https://arxiv.org/abs/1611.08024)
