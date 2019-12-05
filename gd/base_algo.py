import numpy as np


class SN:
    def __init__(self, w_init, b_init):
        self.w = w_init
        self.b = b_init
        self.w_h = []
        self.b_h = []
        self.e_h = []

    def sigmoid(self, x, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        return 1. / (1. + np.exp(-(w * x + b)))

    def error(self, X, Y, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        err = 0
        for x, y in zip(X, Y):
            err += 0.5 * (self.sigmoid(x, w, b) - y) ** 2
        return err

    def grad_w(self, x, y, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        y_pred = self.sigmoid(x, w, b)
        return (y_pred - y) * y_pred * (1 - y_pred) * x

    def grad_b(self, x, y, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        y_pred = self.sigmoid(x, w, b)
        return (y_pred - y) * y_pred * (1 - y_pred)

    def clear(self, x_samples, y_samples):
        self.w_h = []
        self.b_h = []
        self.e_h = []
        self.X = x_samples
        self.Y = y_samples

    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    def append_log(self):
        self.w_h.append(self.w)
        self.b_h.append(self.b)
        self.e_h.append(self.error(self.X, self.Y))
