import tensorflow as tf

from cnn_encoder import build_cnn_encoder
from sequence_model import build_sequence_model
from decoder import build_decoder

VOCAB_SIZE = 20


def build_hmer_model():

    image_input = tf.keras.layers.Input(
        shape=(128, 128, 1),
        name="image_input"
    )

    # CNN
    cnn_encoder = build_cnn_encoder()
    cnn_features = cnn_encoder(image_input)

    # BiLSTM
    sequence_model = build_sequence_model()
    sequence_features = sequence_model(cnn_features)

    # Reduce 1024 time steps to 20
    sequence_features = tf.keras.layers.AveragePooling1D(
        pool_size=49,
        strides=49
    )(sequence_features)

    # Decoder
    decoder = build_decoder(VOCAB_SIZE)
    output = decoder(sequence_features)

    model = tf.keras.Model(
        inputs=image_input,
        outputs=output,
        name="HMER_Model"
    )

    return model


if __name__ == "__main__":
    model = build_hmer_model()
    model.summary()