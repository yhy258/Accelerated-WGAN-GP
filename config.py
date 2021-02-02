

class model_config:
    latent_dim = 128
    image_shape = (28,28,1)
    critic_iter = 5

class train_config:
    epochs = 25
    batch_size = 32
    gp_gamma = 10.0