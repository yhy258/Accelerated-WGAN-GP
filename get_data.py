from config import *
from keras import datasets
import tensorflow as tf

(train_x, train_y), (test_x, test_y) = datasets.fashion_mnist.load_data()

train_x = train_x.reshape(train_x.shape[0], *model_config.image_shape).astype('float32')
train_x = (train_x - 127.5) / 127.5

batch_size = model_config.batch_size

train_x_batched = tf.data.Dataset.from_tensor_slices(train_x).shuffle(len(train_x)).batch(batch_size)