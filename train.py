# from models.ffnn import BrainFFNN
# from models.vgg import BrainVGG

# ffnn = BrainFFNN()
# hist = ffnn.train(epochs=12)
# ffnn.eval()
# ffnn.save()
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))