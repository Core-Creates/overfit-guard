"""
Example: Using overfit-guard with scikit-learn

This example demonstrates how to integrate overfit-guard with scikit-learn models.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from overfit_guard.integrations.sklearn import create_sklearn_monitor


def main():
    # Set random seed
    np.random.seed(42)

    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Dataset shape:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")

    # Create scikit-learn model (using MLPClassifier for iterative training)
    model = MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=1,  # We'll train iteratively
        warm_start=True,
        random_state=42
    )

    # Create overfit monitor
    monitor = create_sklearn_monitor(
        config={
            'log_level': 'INFO'
        },
        metric_name='accuracy',
        higher_is_better=True,
        verbose=True
    )

    # Manual training loop with monitoring
    print("\nStarting training with overfit-guard monitoring...\n")

    n_epochs = 50

    for epoch in range(n_epochs):
        # Train for one iteration
        model.fit(X_train, y_train)

        # Get predictions and scores
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))

        # Print metrics
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Val Accuracy:   {val_acc:.4f}")

        # Check for overfitting
        results = monitor.check_iteration(
            iteration=epoch,
            train_metrics={'accuracy': train_acc},
            val_metrics={'accuracy': val_acc},
            model=model
        )

        # Display overfitting status
        if results['is_overfitting']:
            print(f"  ⚠️  Overfitting detected (Severity: {results['max_severity'].name})")

            # Show recommendations
            if results['detections']:
                for detection in results['detections']:
                    det_result = detection['result']
                    if det_result.recommendations:
                        print(f"  💡 Recommendations:")
                        for rec in det_result.recommendations:
                            print(f"     - {rec}")

        # Check for early stopping recommendation
        if monitor.should_stop:
            print(f"\n⏹️  Early stopping recommended at epoch {epoch+1}")
            break

        print()

    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    summary = monitor.monitor.get_summary()
    print(f"Total epochs: {epoch + 1}")
    print(f"Overfitting detected: {summary['overfitting_detected']} times")
    print(f"Final train accuracy: {train_acc:.4f}")
    print(f"Final val accuracy: {val_acc:.4f}")

    # Example: Using cross-validation based detection
    print("\n" + "="*60)
    print("Cross-Validation Analysis")
    print("="*60)

    # Create a fresh model for CV analysis
    cv_model = MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=100,
        random_state=42
    )

    # Run CV-based overfitting detection
    cv_results = monitor.check_cross_validation(
        model=cv_model,
        X=X_train,
        y=y_train,
        cv=5
    )

    print(f"Overfitting detected via CV: {cv_results['is_overfitting']}")
    if cv_results['is_overfitting']:
        print(f"Severity: {cv_results['max_severity'].name}")


if __name__ == "__main__":
    main()
