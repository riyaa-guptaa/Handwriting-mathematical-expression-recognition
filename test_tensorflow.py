# model/cnn_encoder


import tensorflow as tf

layers = tf.keras.layers


def build_cnn_encoder():

    inputs = layers.Input(
        shape=(128, 128, 1)
    )



    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)


    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)


    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)


    # Convert feature map into sequence

    x = layers.Reshape(
        (-1, 128)
    )(x)


    model = tf.keras.Model(
        inputs,
        x,
        name="CNN_Encoder"
    )

    return model