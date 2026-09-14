"""Testes unitários para validação das regras de negócio."""

import joblib
from sklearn.metrics import accuracy_score, classification_report

from src.data.loader import (
    load_fashion_mnist,
    normalize_images,
    flatten_images,
)

model = joblib.load("models/svm.joblib")

(x_train, y_train), (x_test, y_test) = load_fashion_mnist()

# Mesma transformação usada no treinamento
x_test = normalize_images(x_test)
x_test_flat = flatten_images(x_test)

pred = model.predict(x_test_flat[:5])

print("Predições:", pred)
print("Reais:     ", y_test[:5])

pred = model.predict(x_test_flat)

print("Accuracy:", accuracy_score(y_test, pred))
print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)