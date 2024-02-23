import tensorflow as tf
from data.data import get_dataset

def make_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, input_shape=(196608,), activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(4, activation="softmax")
        ]
    )

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_test(model, epochs=10):
    train_ds, test_ds = get_dataset()
    model.fit(train_ds, epochs=epochs)
    model.evaluate(test_ds)
