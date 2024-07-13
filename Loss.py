import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def binary_crossentropy(y_true, y_pred):
    # Ajout d'une petite valeur epsilon pour éviter log(0)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_prime(y_true, y_pred):
    # Ajout d'une petite valeur epsilon pour éviter division par zéro
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)