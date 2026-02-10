import click
import yaml
import torch
from src.model import EEGNet
from src.data import get_dataloaders, generate_synthetic_data
from src.train import Trainer
from src.utils import plot_training_history, plot_confusion_matrix, get_performance_report
import os

@click.group()
def cli():
    """EEGNet: Startup-grade implementation for EEG classification."""
    pass

@cli.command()
@click.option('--config', default='config/base_config.yaml', help='Path to config file.')
@click.option('--output', default='outputs/', help='Directory to save results.')
def train(config, output):
    """Train the EEGNet model."""
    # Load config
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    if not os.path.exists(output):
        os.makedirs(output)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data for demonstration (BCI IV-2a style)
    print("Generating synthetic data for demonstration...")
    train_data, train_labels = generate_synthetic_data(n_samples=500, chans=cfg['model']['chans'], samples=cfg['model']['samples'])
    val_data, val_labels = generate_synthetic_data(n_samples=100, chans=cfg['model']['chans'], samples=cfg['model']['samples'])
    
    train_loader, val_loader = get_dataloaders(train_data, train_labels, val_data, val_labels, batch_size=cfg['training']['batch_size'])
    
    # Initialize model
    model = EEGNet(
        chans=cfg['model']['chans'],
        samples=cfg['model']['samples'],
        dropout_rate=cfg['model']['dropout'],
        f1=cfg['model']['f1'],
        d=cfg['model']['d'],
        f2=cfg['model']['f2'],
        nb_classes=cfg['model']['nb_classes']
    )
    
    trainer = Trainer(model, train_loader, val_loader, cfg['training'], device=device)
    
    print("Starting training...")
    best_acc = trainer.train(epochs=cfg['training']['epochs'], checkpoint_path=os.path.join(output, 'best_model.pth'))
    
    print(f"Training completed. Best Val Acc: {best_acc:.2f}%")

@cli.command()
@click.option('--model_path', required=True, help='Path to saved model.')
@click.option('--config', default='config/base_config.yaml', help='Path to config file.')
def evaluate(model_path, config):
    """Evaluate a trained model."""
    # Load config
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = EEGNet(
        chans=cfg['model']['chans'],
        samples=cfg['model']['samples'],
        nb_classes=cfg['model']['nb_classes']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Generate test data
    test_data, test_labels = generate_synthetic_data(n_samples=100, chans=cfg['model']['chans'], samples=cfg['model']['samples'])
    
    test_inputs = torch.from_numpy(test_data).float().to(device)
    with torch.no_grad():
        outputs = model(test_inputs)
        _, predicted = outputs.max(1)
        
    y_true = test_labels
    y_pred = predicted.cpu().numpy()
    
    report = get_performance_report(y_true, y_pred, [f"Class {i}" for i in range(cfg['model']['nb_classes'])])
    print("Classification Report:")
    print(report)
    
    plot_confusion_matrix(y_true, y_pred, [f"Class {i}" for i in range(cfg['model']['nb_classes'])])
    print("Confusion matrix saved to plots/confusion_matrix.png")

if __name__ == "__main__":
    cli()
