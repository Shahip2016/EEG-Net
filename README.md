# EEGNet-Startup

A professional, startup-grade implementation of the **EEGNet** architecture (Lawhern et al. 2018) for EEG signal classification.

## 🚀 Features

- **Core Architecture**: Faithful PyTorch implementation of EEGNet for multiple BCI paradigms.
- **Compact & Efficient**: Optimized for low parameter counts.
- **Professional CLI**: Easily train and evaluate models with detailed configuration.
- **Visualization**: Automatic plotting of training history and confusion matrices.
- **Config-Driven**: All hyperparameters managed via YAML.

## 🛠️ Installation

```bash
git clone git@github.com:Shahip2016/EEG-Net.git
cd EEG-Net
pip install -r requirements.txt
```

## 📖 Usage

### Training
To train the model with the default configuration:
```bash
python main.py train --config config/base_config.yaml --output outputs/
```

### Evaluation
To evaluate a trained model:
```bash
python main.py evaluate --model_path outputs/best_model.pth --config config/base_config.yaml
```

## 📊 Architecture Details
EEGNet uses a combination of:
1. **Temporal Convolution**: Learns frequency-specific features.
2. **Depthwise Convolution**: Captures spatial filters (spatial filtering).
3. **Separable Convolution**: Minimizes parameters while refining features.

## 📚 References
- Lawhern, V. J., et al. (2018). "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces." *Journal of Neural Engineering*. [Link](https://arxiv.org/abs/1611.08024)
