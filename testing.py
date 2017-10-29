from PIL import Image
from pycf.data_providers import CIFAR10DataProvider

train_data = CIFAR10DataProvider()
batch = None

for input_batch, target_batch in train_data:
    batch = input_batch

img = Image.fromarray(batch[0, :, :, :], 'RGB')
img.save('images/my.png')
