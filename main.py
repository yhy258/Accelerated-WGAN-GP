import tensorflow as tf
from keras.utils import Progbar
from get_data import *
from train import train_step
import numpy as np
from config import *
epochs = train_config.epochs

for epoch in range(epochs):
  tf.print("{}/{} epoch".format(epoch+1, epochs))
  pbar = Progbar(target = 60000, unit_name = "WGAN_GP")
  prev_dis_loss,prev_gen_loss = 0,0
  for x in train_x_batched:
    dis_loss, gen_loss = train_step(x,prev_dis_loss,prev_gen_loss)
    prev_dis_loss = dis_loss
    prev_gen_loss = gen_loss
    values=[("Critic Loss", np.round(dis_loss.numpy(),4)), ("Generator Loss", np.round(gen_loss.numpy(),4))]
    pbar.add(x.shape[0],values=values)