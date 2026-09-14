
from tensorflow import keras


def build_cnn(
    input_shape: tuple[int, ...] = (28, 28, 1),
    n_classes: int = 10,
    filters_1: int = 32,
    filters_2: int = 64,
    dense_units: int = 128,
    dropout_rate: float = 0.30,
) -> keras.Model:
    """
    Constrói e compila a CNN baseline para Fashion-MNIST.

    Parameters
    ----------
    input_shape : tuple
        Dimensão das imagens de entrada.

    n_classes : int
        Número de classes de saída.

    filters_1 : int
        Número de filtros da primeira camada convolucional.

    filters_2 : int
        Número de filtros da segunda camada convolucional.

    dense_units : int
        Número de neurônios da camada densa.

    dropout_rate : float
        Taxa de dropout.

    Returns
    -------
    keras.Model
        Modelo CNN compilado.
    """

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),

            keras.layers.Conv2D(
                filters_1,
                kernel_size=3,
                activation="relu",
            ),

            keras.layers.MaxPooling2D(),

            keras.layers.Conv2D(
                filters_2,
                kernel_size=3,
                activation="relu",
            ),

            keras.layers.MaxPooling2D(),

            keras.layers.Flatten(),

            keras.layers.Dense(
                dense_units,
                activation="relu",
            ),

            keras.layers.Dropout(dropout_rate),

            keras.layers.Dense(
                n_classes,
                activation="softmax",
            ),
        ],
        name="fashion_mnist_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model