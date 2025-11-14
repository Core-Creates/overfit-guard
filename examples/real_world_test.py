"""
Real-world Dataset Test: Breast Cancer Detection
Uses Wisconsin Breast Cancer dataset from sklearn
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from overfit_guard.integrations.pytorch import create_pytorch_monitor


class BreastCancerNet(nn.Module):
    """Neural network for breast cancer classification."""

    def __init__(self, input_size=30):
        super(BreastCancerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def plot_training_history(history, with_guard=True):
    """Plot training history."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss vs. Epoch' + (' (With Overfit Guard)' if with_guard else ' (Without Overfit Guard)'), fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy vs. Epoch' + (' (With Overfit Guard)' if with_guard else ' (Without Overfit Guard)'), fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'training_history_{"with" if with_guard else "without"}_guard.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as: training_history_{'with' if with_guard else 'without'}_guard.png")
    plt.close()


def run_experiment(use_guard=True, num_epochs=100):
    """Run training experiment."""
    print(f"\n{'='*80}")
    print(f"Running Experiment: {'WITH' if use_guard else 'WITHOUT'} Overfit Guard")
    print(f"{'='*80}\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and prepare data
    print("Loading Wisconsin Breast Cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target

    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)} (0=malignant, 1=benign)")
    print(f"Class distribution: {np.bincount(y)}")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples\n")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create model, criterion, and optimizer
    model = BreastCancerNet(input_size=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}\n")

    # Create overfit monitor if enabled
    monitor = None
    if use_guard:
        monitor = create_pytorch_monitor(
            model=model,
            optimizer=optimizer,
            config={
                'auto_correct': True,
                'min_severity_for_correction': 'MODERATE',
                'correction_cooldown': 5
            },
            auto_correct=True
        )
        print("Overfit Guard ENABLED\n")
    else:
        print("Overfit Guard DISABLED\n")

    # Training loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    overfitting_events = []

    print("Starting training...\n")
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Check for overfitting with monitor
        if monitor:
            results = monitor.on_epoch_end(
                epoch=epoch,
                model=model,
                train_metrics={'loss': train_loss, 'accuracy': train_acc},
                val_metrics={'loss': val_loss, 'accuracy': val_acc}
            )

            if results['is_overfitting']:
                overfitting_events.append({
                    'epoch': epoch,
                    'severity': results['max_severity'].name,
                    'corrections': len(results['corrections'])
                })

                if (epoch + 1) % 10 == 0 and results['corrections']:
                    print(f"  🔧 Applied {len(results['corrections'])} correction(s)")

            # Check for early stopping
            if monitor.should_stop:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print()

    # Final evaluation on test set
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")

    # Print summary
    print("="*80)
    print("Training Summary")
    print("="*80)
    print(f"Total epochs: {len(history['train_loss'])}")
    print(f"Best train accuracy: {max(history['train_acc']):.4f}")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
    print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Train-Val gap: {(history['train_acc'][-1] - history['val_acc'][-1]):.4f}")

    if monitor:
        summary = monitor.monitor.get_summary()
        print(f"\nOverfit Guard Statistics:")
        print(f"  Overfitting detected: {summary['overfitting_detected']} times")
        print(f"  Corrections applied: {summary['corrections_applied']}")
        print(f"  Overfitting rate: {summary['overfitting_rate']:.2%}")

    print()

    # Plot training history
    plot_training_history(history, with_guard=use_guard)

    return {
        'history': history,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'overfitting_events': overfitting_events if monitor else [],
        'monitor_summary': monitor.monitor.get_summary() if monitor else None
    }


def main():
    """Main function to run comparative experiments."""
    print("\n" + "="*80)
    print("OVERFIT GUARD - REAL-WORLD EVALUATION")
    print("Dataset: Wisconsin Breast Cancer (Binary Classification)")
    print("="*80)

    # Run experiment without guard
    results_without = run_experiment(use_guard=False, num_epochs=100)

    # Run experiment with guard
    results_with = run_experiment(use_guard=True, num_epochs=100)

    # Comparative analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    print("\nWithout Overfit Guard:")
    print(f"  Final Train Acc: {results_without['history']['train_acc'][-1]:.4f}")
    print(f"  Final Val Acc: {results_without['history']['val_acc'][-1]:.4f}")
    print(f"  Test Acc: {results_without['test_acc']:.4f}")
    print(f"  Train-Val Gap: {(results_without['history']['train_acc'][-1] - results_without['history']['val_acc'][-1]):.4f}")

    print("\nWith Overfit Guard:")
    print(f"  Final Train Acc: {results_with['history']['train_acc'][-1]:.4f}")
    print(f"  Final Val Acc: {results_with['history']['val_acc'][-1]:.4f}")
    print(f"  Test Acc: {results_with['test_acc']:.4f}")
    print(f"  Train-Val Gap: {(results_with['history']['train_acc'][-1] - results_with['history']['val_acc'][-1]):.4f}")
    print(f"  Overfitting Events: {len(results_with['overfitting_events'])}")
    print(f"  Corrections Applied: {results_with['monitor_summary']['corrections_applied']}")

    print("\nImprovement:")
    test_acc_improvement = (results_with['test_acc'] - results_without['test_acc']) * 100
    gap_reduction = (results_without['history']['train_acc'][-1] - results_without['history']['val_acc'][-1]) - \
                    (results_with['history']['train_acc'][-1] - results_with['history']['val_acc'][-1])

    print(f"  Test Accuracy: {test_acc_improvement:+.2f}% {'✅' if test_acc_improvement > 0 else '❌'}")
    print(f"  Train-Val Gap Reduction: {gap_reduction:+.4f} {'✅' if gap_reduction > 0 else '❌'}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if test_acc_improvement > 0:
        print("✅ Overfit Guard IMPROVED test set performance")
    else:
        print("⚠️  Overfit Guard did not improve test set performance")

    if gap_reduction > 0:
        print("✅ Overfit Guard REDUCED train-validation gap")
    else:
        print("⚠️  Overfit Guard did not reduce train-validation gap")

    print("\nNote: Check generated plots for detailed training dynamics.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
