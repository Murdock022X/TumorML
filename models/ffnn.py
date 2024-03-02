import tensorflow as tf
from data.data import get_dataset

class BrainFFNN:

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

        self.train_ds, self.test_ds = get_dataset()
    
    def train(self, epochs=10):
        self.model.fit(self.train_ds, epochs=epochs)

    def eval(self):
        self.model.evaluate(self.test_ds)

    def save(self):
        self.model.save('ffnn-weights.keras')

    def load():
        self.model.load_weights('ffnn-weights.keras')
