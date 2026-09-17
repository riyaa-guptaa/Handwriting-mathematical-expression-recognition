import tensorflow as tf

layers = tf.keras.layers


def build_sequence_model():

    inputs = layers.Input(shape=(None, 128))

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True)
    )(inputs)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True)
    )(x)

    model = tf.keras.Model(
        inputs,
        x,
        name="BiLSTM_Encoder"
    )

    return model


if __name__ == "__main__":

    model = build_sequence_model()

    model.summary()