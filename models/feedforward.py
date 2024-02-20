import tensorflow as tf

class FeedForward:

    def __init__(self):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, input_shape=(196608,), activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(4, activation="softmax")
            ]
        )

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit(self, dataset):
        self.model.fit(dataset, epochs=12)

    def test(self, dataset):
        self.model.evaluate(dataset)
