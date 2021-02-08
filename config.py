

import os

nb_train_samples = 50000
nb_test_samples = 300000

img_size = 32
img_channel = 3
img_shape =(img_size,img_size,img_channel)

def root_path():
    return os.path.dirname(__file__)

def checkpoint_path():
    return os.path.join(root_path(),"checkpoint")

print(root_path())