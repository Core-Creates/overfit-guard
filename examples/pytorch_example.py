"""
Example: Using overfit-guard with PyTorch

This example demonstrates how to integrate overfit-guard with a PyTorch training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from overfit_guard.integrations.pytorch import create_pytorch_monitor


# Simple neural network for demonstration
class SimpleNet(nn.Module):
    def __init__(self, input_size=20, hidden_size=50, output_size=2):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
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


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create synthetic dataset
    torch.manual_seed(42)
    X_train = torch.randn(1000, 20)
    y_train = torch.randint(0, 2, (1000,))
    X_val = torch.randn(200, 20)
    y_val = torch.randint(0, 2, (200,))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model, criterion, and optimizer
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create overfit monitor with auto-correction
    monitor_callback = create_pytorch_monitor(
        model=model,
        optimizer=optimizer,
        config={
            'auto_correct': True,
            'min_severity_for_correction': 'MODERATE',
            'correction_cooldown': 5
        },
        auto_correct=True
    )

    # Training loop
    num_epochs = 50
    print("\nStarting training with overfit-guard monitoring...\n")

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Check for overfitting
        results = monitor_callback.on_epoch_end(
            epoch=epoch,
            model=model,
            train_metrics={'loss': train_loss, 'accuracy': train_acc},
            val_metrics={'loss': val_loss, 'accuracy': val_acc}
        )

        # Display overfitting status
        if results['is_overfitting']:
            print(f"  ⚠️  Overfitting detected (Severity: {results['max_severity'].name})")

            # Show applied corrections
            if results['corrections']:
                print(f"  🔧 Applied {len(results['corrections'])} correction(s):")
                for corr in results['corrections']:
                    print(f"     - {corr['corrector']}: {corr['result'].message}")

        # Check for early stopping
        if monitor_callback.should_stop:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
            break

        print()

    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    summary = monitor_callback.monitor.get_summary()
    print(f"Total epochs: {epoch + 1}")
    print(f"Overfitting detected: {summary['overfitting_detected']} times")
    print(f"Corrections applied: {summary['corrections_applied']}")
    print(f"Overfitting rate: {summary['overfitting_rate']:.2%}")


if __name__ == "__main__":
    main()
