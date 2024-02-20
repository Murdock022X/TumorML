import tensorflow as tf
from pathlib import Path

def get_dataset():
    direc = Path(__file__).parent / Path('dataset1')
    return tf.keras.utils.image_dataset_from_directory(str(direc / Path('Training')), batch_size=32), tf.keras.utils.image_dataset_from_directory(str(direc / Path('Testing')), batch_size=32)
