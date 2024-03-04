import tensorflow as tf
from pathlib import Path

direc = Path(__file__).parent / Path('dataset1')

def get_train_dataset():
    return tf.keras.utils.image_dataset_from_directory(str(direc / Path('Training')), batch_size=32, validation_split=0.2, subset="training", color_mode='grayscale', seed=1)

def get_val_dataset():
    return tf.keras.utils.image_dataset_from_directory(str(direc / Path('Training')), batch_size=32, validation_split=0.2, subset="validation", color_mode='grayscale', seed=1)

def get_test_dataset():
    return tf.keras.utils.image_dataset_from_directory(str(direc / Path('Testing')), batch_size=32, color_mode='grayscale')