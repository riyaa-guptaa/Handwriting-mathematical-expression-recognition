import tensorflow as tf

layers = tf.keras.layers


def build_decoder(vocab_size):

    inputs = layers.Input(shape=(None, 256))

    x = layers.LSTM(
        256,
        return_sequences=True
    )(inputs)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        vocab_size,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="HMER_Decoder"
    )

    return model


if __name__ == "__main__":

    vocab_size = 20

    model = build_decoder(vocab_size)

    model.summary()