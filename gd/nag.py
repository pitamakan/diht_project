from . import base_algo


class NAG(base_algo.SN):
    def __init__(self, w_init, b_init):
        super().__init__(w_init, b_init)

    def fit(self, x_samples, y_samples, epochs=100, eta=0.01, gamma=0.9):
        self.clear(x_samples, y_samples)

        v_w, v_b = 0, 0
        for i in range(epochs):
            dw, db = 0, 0
            v_w = gamma * v_w
            v_b = gamma * v_b
            for x, y in zip(x_samples, y_samples):
                dw += self.grad_w(x, y, self.w - v_w, self.b - v_b)
                db += self.grad_b(x, y, self.w - v_w, self.b - v_b)
            v_w = v_w + eta * dw
            v_b = v_b + eta * db
            self.w = self.w - v_w
            self.b = self.b - v_b
            self.append_log()
