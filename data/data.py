import tensorflow as tf
from pathlib import Path

cat_root = Path(__file__).parent / Path('cat_data')
bin_root = Path(__file__).parent / Path('bin_data')

def get_train_dataset():
    return tf.keras.utils.image_dataset_from_directory(str(cat_root / Path('Training')), batch_size=32, validation_split=0.2, subset="training", color_mode='grayscale', seed=1)

def get_val_dataset():
    return tf.keras.utils.image_dataset_from_directory(str(cat_root / Path('Training')), batch_size=32, validation_split=0.2, subset="validation", color_mode='grayscale', seed=1)

def get_test_dataset():
    return tf.keras.utils.image_dataset_from_directory(str(cat_root / Path('Testing')), batch_size=32, color_mode='grayscale')

def get_bin_train_dataset():
    return tf.keras.utils.image_dataset_from_directory(str(bin_root / Path('Training')), batch_size=32, validation_split=0.2, subset="training", color_mode='grayscale', seed=1, label_mode='binary')

def get_bin_val_dataset():
    return tf.keras.utils.image_dataset_from_directory(str(bin_root / Path('Training')), batch_size=32, validation_split=0.2, subset="validation", color_mode='grayscale', seed=1, label_mode='binary')

def get_bin_test_dataset():
    return tf.keras.utils.image_dataset_from_directory(str(bin_root / Path('Testing')), batch_size=32, color_mode='grayscale', label_mode='binary')

def make_bin_directories():
    for dir in ['Training', 'Testing']:
        cat_dir = cat_root / Path(dir)
        bin_dir = bin_root / Path(dir)

        tumor = set(['glioma', 'meningioma', 'pituitary'])
        
        for f in cat_dir.rglob('*.jpg'):
            cat_f = cat_dir / f
            bytes = cat_f.read_bytes()

            bin_sub = None
            
            if str(f.parent.name) in tumor:
                bin_sub = bin_dir / Path('tumor')
            else:
                bin_sub = bin_dir / Path('notumor')
                
            bin_sub.mkdir(parents=True, exist_ok=True)

            bin_f = bin_sub / Path(f.name)

            bin_f.touch()

            bin_f.write_bytes(bytes)
        
    