import click
import yaml
import torch
import numpy as np
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
@click.option('--epochs', type=int, help='Override number of epochs.')
@click.option('--lr', type=float, help='Override learning rate.')
@click.option('--seed', type=int, help='Set random seed for reproducibility.')
def train(config, output, epochs, lr, seed):
    """Train the EEGNet model with robust configuration and seed control."""
    # Load config
    if not os.path.exists(config):
        print(f"Error: Config file not found at {config}")
        return
        
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override config with CLI arguments if provided
    if epochs: cfg['training']['epochs'] = epochs
    if lr: cfg['training']['lr'] = lr
    if seed: cfg['training']['seed'] = seed
    
    # Set seeds for reproducibility
    target_seed = cfg['training'].get('seed', 42)
    torch.manual_seed(target_seed)
    np.random.seed(target_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(target_seed)
    
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Seed: {target_seed}")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    train_data, train_labels = generate_synthetic_data(
        n_samples=500, 
        chans=cfg['model']['chans'], 
        samples=cfg['model']['samples']
    )
    val_data, val_labels = generate_synthetic_data(
        n_samples=100, 
        chans=cfg['model']['chans'], 
        samples=cfg['model']['samples']
    )
    
    train_loader, val_loader = get_dataloaders(
        train_data, train_labels, 
        val_data, val_labels, 
        batch_size=cfg['training']['batch_size']
    )
    
    # Initialize model
    model = EEGNet(
        chans=cfg['model']['chans'],
        samples=cfg['model']['samples'],
        dropout_rate=cfg['model']['dropout'],
        kern_length=cfg['model'].get('kern_length', 64),
        f1=cfg['model']['f1'],
        d=cfg['model']['d'],
        f2=cfg['model']['f2'],
        nb_classes=cfg['model']['nb_classes']
    )
    
    trainer = Trainer(model, train_loader, val_loader, cfg['training'], device=device)
    
    print("Starting training...")
    history = trainer.train(epochs=cfg['training']['epochs'], checkpoint_path=os.path.join(output, 'best_model.pth'))
    
    print(f"Training completed. Best Val Acc: {trainer.best_val_acc:.2f}%")
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history, save_path=os.path.join(output, 'training_history.png'))
    print(f"Training history plot saved to {os.path.join(output, 'training_history.png')}")

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
