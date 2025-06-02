import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape, conv_filters=[16, 32], kernel_sizes=[(5, 5), (3, 3)], conv_activations=['relu', 'relu'], dense_activation='sigmoid') -> tf.keras.Model:
    """
    Defines the architecture of a CNN using Keras Sequential API.
    Args: input_shape (tuple): Shape of the input image, default is (256, 256, 3) for RGB images.
    Returns: tf.keras.Model: Uncompiled CNN model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    for filters, kernel_size, activation in zip(conv_filters, kernel_sizes, conv_activations):
        model.add(layers.Conv2D(filters, kernel_size, activation=activation))
        model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation=dense_activation))
    return model

def compile_model(model, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) -> tf.keras.Model:
    """
    Compiles the model with optimizer, loss, and evaluation metrics.
    Args: model (tf.keras.Model): The model to be compiled.
    Returns: tf.keras.Model: Compiled Keras model ready for training.
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
