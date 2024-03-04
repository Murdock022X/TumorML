import tensorflow as tf
from data.data import get_train_dataset, get_val_dataset, get_test_dataset, get_bin_train_dataset, get_bin_val_dataset, get_bin_test_dataset

class BrainFFNN:

    def __init__(self, binary=False):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, input_shape=(65536,), activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(4, activation="softmax")
            ]
        )

        self.model.compile(optimizer='adam', 
                           loss='sparse_categorical_crossentropy', 
                           metrics=['accuracy'])

        if binary:
            self.train_ds = get_bin_train_dataset()
            self.val_ds = get_bin_val_dataset()
            self.test_ds = get_bin_test_dataset()
            
        else:
            self.train_ds = get_train_dataset()
            self.val_ds = get_val_dataset()
            self.test_ds = get_test_dataset()
    
    def train(self, epochs=10):
        return self.model.fit(self.train_ds, epochs=epochs, validation_data=self.val_ds)

    def eval(self):
        self.model.evaluate(self.test_ds)

    def save(self):
        self.model.save('ffnn-weights.keras')

    def load(self):
        self.model.load_weights('ffnn-weights.keras')
