from . import base_algo

import numpy as np


class Adam(base_algo.SN):
    def __init__(self, w_init, b_init):
        super().__init__(w_init, b_init)

    def fit(self, x_samples, y_samples, epochs=100, eta=0.01, eps=1e-8, beta1=0.9, beta2=0.9):
        self.clear(x_samples, y_samples)

        v_w, v_b = 0, 0
        m_w, m_b = 0, 0
        num_updates = 0
        for i in range(epochs):
            dw, db = 0, 0
            for x, y in zip(x_samples, y_samples):
                dw = self.grad_w(x, y)
                db = self.grad_b(x, y)
                num_updates += 1
                m_w = beta1 * m_w + (1 - beta1) * dw
                m_b = beta1 * m_b + (1 - beta1) * db
                v_w = beta2 * v_w + (1 - beta2) * dw ** 2
                v_b = beta2 * v_b + (1 - beta2) * db ** 2
                m_w_c = m_w / (1 - np.power(beta1, num_updates))
                m_b_c = m_b / (1 - np.power(beta1, num_updates))
                v_w_c = v_w / (1 - np.power(beta2, num_updates))
                v_b_c = v_b / (1 - np.power(beta2, num_updates))
                self.w -= (eta / np.sqrt(v_w_c) + eps) * m_w_c
                self.b -= (eta / np.sqrt(v_b_c) + eps) * m_b_c
                self.append_log()
