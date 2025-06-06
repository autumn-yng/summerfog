import pytest
import tensorflow as tf
from src.model.model import build_model, compile_model
from src.data import load_dataset, scale_dataset, split_dataset

@pytest.fixture
def dataset():
    ds = load_dataset("data", class_names=["no_fog", "fog"], batch_size=5)
    ds = scale_dataset(ds)
    return split_dataset(ds)  # Returns train, val, test datasets

def test_training_runs(dataset):
    train_ds, val_ds, _ = dataset

    # Build a minimal model for testing
    model = build_model(
        input_shape=(256, 256, 3),  # Match image shape
        conv_filters=[8],          # Use smaller filter for faster test
        kernel_sizes=[(3, 3)],
        conv_activations=["relu"],
        dense_activation="sigmoid"
    )

    # Compile the model with standard settings
    model = compile_model(model, optimizer="adam", loss="binary_crossentropy")

    # Run training for 1 epoch to verify pipeline works without error
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        verbose=0  # Suppress output for test clarity
    )

    # Ensure history object is returned and contains 'loss'
    assert "loss" in history.history
    assert "val_loss" in history.history
