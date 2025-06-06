import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import os
import pytest

@pytest.mark.skipif(
    not os.path.exists("saved_models/best_model.keras") or not os.path.exists("data/test_data"),
    reason="Model or test data not found. Run train.py first."
)
def test_evaluate_metrics():
    # Load trained model and test dataset
    model = tf.keras.models.load_model("models/best_model.keras")
    test_ds = tf.data.Dataset.load("data/test_data")

    # Initialize metrics
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    # Evaluate model predictions on test data
    for X, y in test_ds:
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)

    # Get final metric values
    precision = pre.result().numpy()
    recall = re.result().numpy()
    accuracy = acc.result().numpy()

    # Check metrics are in valid range
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= accuracy <= 1
