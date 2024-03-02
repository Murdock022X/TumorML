import tensorflow as tf
from data.data import get_dataset

class BrainVGG:

    def __init__(self):
        self.model = tf.keras.applications.vgg19.VGG19(
            weights=None,
            input_shape=(256,256,3),
            classes=4
        )

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.train_ds, self.test_ds = get_dataset()
    
    def train(self, epochs=10):
        self.model.fit(self.train_ds, epochs=epochs)

    def eval(self):
        self.model.evaluate(self.test_ds)

    def save(self):
        self.model.save('vgg-weights.keras')

    def load(self):
        self.model.load_weights('vgg-weights.keras')
