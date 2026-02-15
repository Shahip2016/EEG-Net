import click
import yaml
import torch
import numpy as np
from src.model import EEGNet
from src.data import get_dataloaders, generate_synthetic_data, get_augmentation_pipeline
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
@click.option('--augment', is_flag=True, help='Enable data augmentation.')
@click.option('--noise_std', type=float, default=0.01, help='Gaussian noise standard deviation.')
@click.option('--drop_rate', type=float, default=0.1, help='Channel dropout rate.')
@click.option('--wandb', is_flag=True, help='Enable Weights & Biases logging.')
@click.option('--wandb_project', default='EEG-Net', help='W&B project name.')
def train(config, output, epochs, lr, seed, augment, noise_std, drop_rate, wandb, wandb_project):
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
    
    cfg['training']['use_wandb'] = wandb
    cfg['training']['wandb_project'] = wandb_project
    
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
    
    transform = None
    if augment:
        print(f"Data augmentation enabled (noise_std={noise_std}, drop_rate={drop_rate})")
        transform = get_augmentation_pipeline(noise_std=noise_std, drop_rate=drop_rate)
        
    train_loader, val_loader = get_dataloaders(
        train_data, train_labels, 
        val_data, val_labels, 
        batch_size=cfg['training']['batch_size'],
        transform=transform
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
    
    # Save report
    report_path = os.path.join(os.path.dirname(model_path), 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")
    
    cm_path = os.path.join(os.path.dirname(model_path), 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, [f"Class {i}" for i in range(cfg['model']['nb_classes'])], save_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")

@cli.command()
@click.option('--model_path', required=True, help='Path to saved PyTorch model.')
@click.option('--config', default='config/base_config.yaml', help='Path to config file.')
@click.option('--output', default='outputs/exported_model.pt', help='Path to save exported TorchScript model.')
def export(model_path, config, output):
    """Export a trained model to TorchScript for production."""
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cpu') # Export for CPU by default
    
    model = EEGNet(
        chans=cfg['model']['chans'],
        samples=cfg['model']['samples'],
        nb_classes=cfg['model']['nb_classes']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Trace the model
    example_input = torch.randn(1, 1, cfg['model']['chans'], cfg['model']['samples'])
    traced_model = torch.jit.trace(model, example_input)
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    traced_model.save(output)
    print(f"Model exported to TorchScript at {output}")

if __name__ == "__main__":
    cli()
