from models.ffnn import BrainFFNN
from models.vgg import BrainVGG

ffnn = BrainFFNN()
hist = ffnn.train(epochs=12)
ffnn.eval()
ffnn.save()