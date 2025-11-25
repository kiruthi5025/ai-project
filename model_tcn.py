import tensorflow as tf
from tensorflow.keras import layers

def build_tcn(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, padding='causal', activation='relu')(inputs)
    x = layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, padding='causal', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    return model
