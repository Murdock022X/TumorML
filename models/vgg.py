import tensorflow as tf
from data.data import get_dataset

def make_model():
    model = tf.keras.applications.vgg19.VGG19(
        weights=None,
        input_shape=(256,256,3),
        classes=4
    )

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_test(model, epochs=10):
    train_ds, test_ds = get_dataset()
    model.fit(train_ds, epochs=epochs)
    model.evaluate(test_ds)