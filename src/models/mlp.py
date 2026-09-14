from tensorflow import keras


def build_mlp(
    input_shape: tuple[int, ...] = (784,),
    n_classes: int = 10,
    hidden_units_1: int = 256,
    hidden_units_2: int = 128,
    dropout_1: float = 0.30,
    dropout_2: float = 0.20,
) -> keras.Model:
    """
    Constrói e compila o modelo MLP utilizado no benchmark
    Fashion-MNIST.
    """

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),

            keras.layers.Dense(
                hidden_units_1,
                activation="relu",
            ),
            keras.layers.Dropout(dropout_1),

            keras.layers.Dense(
                hidden_units_2,
                activation="relu",
            ),
            keras.layers.Dropout(dropout_2),

            keras.layers.Dense(
                n_classes,
                activation="softmax",
            ),
        ],
        name="fashion_mnist_mlp",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model