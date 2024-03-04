import tensorflow as tf
from data.data import get_train_dataset, get_val_dataset, get_test_dataset

class BrainVGG:

    def __init__(self):
        self.model = tf.keras.applications.vgg19.VGG19(
            weights=None,
            input_shape=(256,256,1),
            classes=4
        )

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.train_ds = get_train_dataset()
        self.val_ds = get_val_dataset()
        self.test_ds = get_test_dataset()
    
    def train(self, epochs=10):
        return self.model.fit(self.train_ds, epochs=epochs, validation_data=self.val_ds)

    def eval(self):
        self.model.evaluate(self.test_ds)

    def save(self):
        self.model.save('vgg-weights.keras')

    def load(self):
        self.model.load_weights('vgg-weights.keras')
