import pytest
import tensorflow as tf
from src.model.model import build_model, compile_model

def test_build_model():
    model = build_model(input_shape=(256, 256, 3))
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 256, 256, 3)

def test_compile_model():
    model = build_model((256, 256, 3))
    compiled_model = compile_model(model)
    assert compiled_model.optimizer is not None