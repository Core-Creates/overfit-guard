"""
Example: Using overfit-guard with TensorFlow/Keras

This example demonstrates how to integrate overfit-guard with Keras.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from overfit_guard.integrations.keras import create_keras_monitor


def create_model(input_shape=20, hidden_size=50, output_size=2):
    """Create a simple neural network."""
    model = keras.Sequential([
        layers.Dense(hidden_size, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.2),
        layers.Dense(output_size, activation='softmax')
    ])
    return model


def main():
    # Set random seed
    np.random.seed(42)
    tf.random.set_seed(42)

    print("TensorFlow version:", tf.__version__)

    # Create synthetic dataset
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, 20)
    y_val = np.random.randint(0, 2, 200)

    # Create model
    model = create_model()

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel architecture:")
    model.summary()

    # Create overfit-guard callback with auto-correction
    overfit_callback = create_keras_monitor(
        config={
            'auto_correct': True,
            'min_severity_for_correction': 'MODERATE',
            'correction_cooldown': 5,
            'log_level': 'INFO'
        },
        auto_correct=True,
        verbose=True
    )

    # Additional callbacks
    callbacks = [
        overfit_callback,
        # You can add other callbacks here
    ]

    # Train model
    print("\nStarting training with overfit-guard monitoring...\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    summary = overfit_callback.monitor_obj.get_summary()
    print(f"Total epochs: {len(history.history['loss'])}")
    print(f"Overfitting detected: {summary['overfitting_detected']} times")
    print(f"Corrections applied: {summary['corrections_applied']}")
    print(f"Overfitting rate: {summary['overfitting_rate']:.2%}")

    # Evaluate on validation set
    print("\nFinal evaluation:")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
