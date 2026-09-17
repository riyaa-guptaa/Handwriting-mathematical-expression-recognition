import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
import pandas as pd
import tensorflow as tf

from image_processor import preprocess_image
from expression_model import build_expression_model


LABEL_FILE = "dataset/labels.csv"
IMAGE_FOLDER = "dataset/images"

MODEL_PATH = "models/expression_model.keras"
CLASS_FILE = "models/expression_classes.json"

AUGMENTATIONS_PER_IMAGE = 60
EPOCHS = 60
BATCH_SIZE = 16


# -----------------------------------------
# 1. Read labels.csv
# -----------------------------------------

data = pd.read_csv(LABEL_FILE)

expressions = data["expression"].astype(str).tolist()

# Each complete expression becomes one class
unique_expressions = list(dict.fromkeys(expressions))

expression_to_id = {
    expression: index
    for index, expression in enumerate(unique_expressions)
}

id_to_expression = {
    str(index): expression
    for expression, index in expression_to_id.items()
}

print("\nExpression classes:")

for expression, class_id in expression_to_id.items():
    print(class_id, "->", expression)


# -----------------------------------------
# 2. Data augmentation
# -----------------------------------------

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),
    tf.keras.layers.RandomZoom(0.05)
])


images = []
labels = []


# -----------------------------------------
# 3. Load every original image
# -----------------------------------------

for _, row in data.iterrows():

    image_name = str(row["image"])
    expression = str(row["expression"])

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    try:

        _, _, _, processed = preprocess_image(image_path)

        class_id = expression_to_id[expression]

        # Add original image
        images.append(processed)
        labels.append(class_id)

        # Create augmented images
        image_tensor = tf.expand_dims(
            processed,
            axis=0
        )

        for _ in range(AUGMENTATIONS_PER_IMAGE):

            augmented = augmentation(
                image_tensor,
                training=True
            )

            augmented = augmented[0].numpy()

            images.append(augmented)
            labels.append(class_id)

        print(
            "Loaded:",
            image_name,
            "->",
            expression
        )

    except Exception as error:

        print("Error loading:", image_name)
        print(error)


# -----------------------------------------
# 4. Convert to NumPy arrays
# -----------------------------------------

X = np.array(images, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

print("\nTotal training images:", len(X))
print("Image shape:", X.shape)
print("Number of classes:", len(unique_expressions))


# -----------------------------------------
# 5. Shuffle dataset
# -----------------------------------------

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]


# -----------------------------------------
# 6. Build CNN model
# -----------------------------------------

model = build_expression_model(
    num_classes=len(unique_expressions)
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# -----------------------------------------
# 7. Train model
# -----------------------------------------

print("\nStarting classifier training...\n")

model.fit(
    X,
    y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.15,
    verbose=1
)


# -----------------------------------------
# 8. Save model
# -----------------------------------------

os.makedirs("models", exist_ok=True)

model.save(MODEL_PATH)


# -----------------------------------------
# 9. Save class names
# -----------------------------------------

with open(
    CLASS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        id_to_expression,
        file,
        indent=4
    )


print("\n====================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("====================================")

print("\nModel saved to:")
print(MODEL_PATH)

print("\nClass mapping saved to:")
print(CLASS_FILE)