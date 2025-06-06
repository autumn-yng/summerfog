import pytest
import tensorflow as tf
from unittest import mock
from src import data

@pytest.fixture # This marks the following function as a fixture, meaning it's a reusable setup function. In this case, a dummy dataset will be created and automatically passed into any test function that names dummy_dataset as a parameter.
def dummy_dataset():
    # Simulate 10 image samples with shape (256, 256, 3) and binary labels
    X = tf.random.uniform((10, 256, 256, 3))  # Random images
    y = tf.random.uniform((10,), maxval=2, dtype=tf.int32)  # Random binary labels
    return tf.data.Dataset.from_tensor_slices((X, y)).batch(2)  # Batch size of 2

# Test if the scale_dataset function correctly scales pixel values to [0, 1]
def test_scale_dataset(dummy_dataset):
    scaled = data.scale_dataset(dummy_dataset, factor=255.0)
    for x, y in scaled.take(1):  # Take the first batch
        assert tf.reduce_max(x) <= 1.0  # All values should be <= 1
        assert tf.reduce_min(x) >= 0.0  # All values should be >= 0

# Test the load_dataset function by mocking TensorFlow's image_dataset_from_directory
@mock.patch("tensorflow.keras.utils.image_dataset_from_directory")
def test_load_dataset(mock_image_loader):
    # Create a fake dataset to return
    dummy = tf.data.Dataset.from_tensor_slices((tf.zeros((4, 256, 256, 3)), tf.zeros((4,))))
    mock_image_loader.return_value = dummy.batch(2)

    # Call the load_dataset function with dummy arguments
    dataset = data.load_dataset("fake_path", ["cloud", "no_cloud"], batch_size=2)

    # Check that the result is a tf.data.Dataset object
    assert isinstance(dataset, tf.data.Dataset)

# Test the split_dataset function to ensure it splits into correct proportions
def test_split_dataset(dummy_dataset):
    # Split the dataset: 60% train, 20% validation, 20% test
    train, val, test = data.split_dataset(dummy_dataset, train_frac=0.6, val_frac=0.2)

    total = sum(1 for _ in dummy_dataset)  # Count total number of batches

    # Check that each split contains the expected number of batches
    assert sum(1 for _ in train) == int(total * 0.6)
    assert sum(1 for _ in val) == int(total * 0.2)
    assert sum(1 for _ in test) == total - int(total * 0.6) - int(total * 0.2)