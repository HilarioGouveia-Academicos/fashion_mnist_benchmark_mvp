import numpy as np

from config.settings import RANDOM_STATE


def load_fashion_mnist():
    from tensorflow.keras.datasets import fashion_mnist

    return fashion_mnist.load_data()


def normalize_images(images):
    return images.astype("float32") / 255.0


def flatten_images(images):
    return images.reshape(len(images), -1)


def sample_training_data(x, y, n_samples):
    if n_samples >= len(x):
        return x, y
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(x), n_samples, replace=False)
    return x[idx], y[idx]
