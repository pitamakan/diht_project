from . import base_algo

import numpy as np


class AdaGrad(base_algo.SN):
    def __init__(self, w_init, b_init):
        super().__init__(w_init, b_init)

    def fit(self, x_samples, y_samples, epochs=100, eta=0.01, eps=1e-8):
        self.clear(x_samples, y_samples)

        v_w, v_b = 0, 0
        for i in range(epochs):
            dw, db = 0, 0
            for x, y in zip(x_samples, y_samples):
                dw += self.grad_w(x, y)
                db += self.grad_b(x, y)
            v_w += dw ** 2
            v_b += db ** 2
            self.w -= (eta / np.sqrt(v_w) + eps) * dw
            self.b -= (eta / np.sqrt(v_b) + eps) * db
            self.append_log()
