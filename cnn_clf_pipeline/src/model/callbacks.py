import tensorflow as tf
import os

def get_callbacks(log_dir:str, model_path:str):
    """
    Creates TensorFlow callbacks for model training: TensorBoard and ModelCheckpoint.

    Args:
        log_dir (str): Directory where TensorBoard logs will be saved.
        model_path (str): File path to save the best model.

    Returns:
        list: A list of TensorFlow callback instances.
    """
    # Ensure the log directory exists so TensorBoard can write logs to it
    os.makedirs(log_dir, exist_ok=True)

    # ModelCheckpoint callback:
    # - Saves model weights to 'model_path'
    # - Only saves when validation accuracy improves (best model)
    # - Useful to restore the best performing model after training
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',      # Metric to monitor
        save_best_only=True,         # Only save if val_accuracy improves
        verbose=1                    # Print log when saving
    )

    # TensorBoard callback:
    # - Logs training/validation metrics to disk
    # - Can be visualized with TensorBoard
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir              # Logs saved here
    )

    return [tensorboard_cb, checkpoint_cb]
