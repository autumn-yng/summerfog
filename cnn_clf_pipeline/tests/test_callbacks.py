import tensorflow as tf
import os
import shutil
import pytest
from src.model.callbacks import get_callbacks

@pytest.fixture
def temp_dirs():
    """
    Pytest fixture to create and clean up temporary directories for logs and models.
    """
    log_dir = "temp_logs"
    model_dir = "temp_models"
    model_path = os.path.join(model_dir, "best_model.keras")

    # Create the directories before the test
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

	# yield gives the test function these values
    yield log_dir, model_path

    # Clean up the directories after the test is done
    shutil.rmtree(log_dir)
    shutil.rmtree(model_dir)

def test_get_callbacks_creates_callbacks(temp_dirs):
    """
    Unit test for the get_callbacks() function.
    Verifies that the function returns a list of two TensorFlow callbacks: 
    TensorBoard and ModelCheckpoint, and that they are properly initialized.
    """
    log_dir, model_path = temp_dirs

    # Call the function being tested
    callbacks = get_callbacks(log_dir, model_path)

    # Ensure the function returns a list with exactly two callbacks
    assert isinstance(callbacks, list)
    assert len(callbacks) == 2

    # Extract callbacks
    tensorboard_cb, checkpoint_cb = callbacks

    # Check types of each callback to ensure correct instantiation
    
    assert isinstance(tensorboard_cb, tf.keras.callbacks.TensorBoard)
    assert isinstance(checkpoint_cb, tf.keras.callbacks.ModelCheckpoint)

    # Ensure the file path and log_dir are set correctly
    assert checkpoint_cb.filepath == model_path
    assert tensorboard_cb.log_dir == log_dir
