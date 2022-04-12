"""
Train a neural network on MNIST with numerical differentiation.
"""

import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt



# neural network methods ---------------------------------------

def initialize(size):   
    """
    Initialize a neural network.
    """
    # Weights
    Ws = []
    # biases
    bs = []

    for i, n in enumerate(size[:-1]):
        m = size[i+1]
        W = np.random.randn(n, m) * 1.001
        b = np.zeros(m)
        Ws.append(W)
        bs.append(b)

    return Ws, bs


def predict(Weights, biases, x):
    """
    Calculate the output.
    """
    for i, W in enumerate(Weights):
        b = biases[i]
        x = np.dot(x, W) + b

        if (i != len(Weights) - 1):
            x = sigmoid(x)
        else:
            x = softmax(x)

    return x


def evaluate(Weights, biases, x, t):
    """
    The loss function.
    """
    y = predict(Weights, biases, x)
    e = cross_entropy_error(y, t)
    return e


def differentiate(Weights, biases, x, t):
    """
    Calculate the gradient.
    """
    h = 1e-4

    # derivatives of Weights
    dWs = []
    # derivatives of biases
    dbs = []

    # initialize gradient
    for i, W in enumerate(Weights):
        b = biases[i]
        dWs.append(np.zeros_like(W, float))
        dbs.append(np.zeros_like(b, float))

    for i, W in enumerate(Weights):
        for j, w in enumerate(W):
            for k, w_k in enumerate(w):
                # f(x + h)
                Weights[i][j][k] = w_k + h
                loss1 = evaluate(Weights, biases, x, t)
                # f(x - h)
                Weights[i][j][k] = w_k - h
                loss2 = evaluate(Weights, biases, x, t)
                # derivative
                d = (loss1 - loss2) / (2*h)
                dWs[i][j][k] = d
                # restoration
                Weights[i][j][k] = w_k

    for i, b in enumerate(biases):
        for j, b_j in enumerate(b):
            # f(x + h)
            biases[i][j] = b_j + h
            loss1 = evaluate(Weights, biases, x, t)
            # f(x - h)
            biases[i][j] = b_j - h
            loss2 = evaluate(Weights, biases, x, t)
            # derivative
            d = (loss1 - loss2) / (2*h)
            dbs[i][j] = d
            # restoration
            biases[i][j] = b_j

    return dWs, dbs



# math functions -----------------------------------------------

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def softmax(x):
    e = np.exp(x)
    # note that `x` is a batch
    y = e / np.sum(e, axis=1, keepdims=True)
    return y


def cross_entropy_error(y, t):
    # in case of `np.log(0)`
    d = 1e-7
    # average the error over the batch size
    e = -np.sum(t * np.log(y+d)) / y.shape[0]
    return e



# MNIST --------------------------------------------------------

(train_images, train_labels), _ = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

# one-hot
_ = np.zeros((train_labels.size, 10))

for i, label in enumerate(train_labels):
  _[i][label] = 1

train_labels = _



# main ---------------------------------------------------------

losses = []

train_size = train_images.shape[0]
batch_size = 100
learning_rate = 0.1

Ws, bs = initialize((784, 50, 10))

for i in range(300):
    ks = np.random.choice(train_size, batch_size)
    batch_images = train_images[ks]
    batch_labels = train_labels[ks]

    dWs, dbs = differentiate(Ws, bs, batch_images, batch_labels)

    for j, W in enumerate(Ws):
        Ws[j] = W - learning_rate*dWs[j]

    for j, b in enumerate(bs):
        bs[j] = b - learning_rate*dbs[j]

    loss = evaluate(Ws, bs, batch_images, batch_labels)
    losses.append(loss)
    print(i, loss)

plt.plot(losses)
plt.show()
