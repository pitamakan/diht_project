from . import base_algo


class BaseGD(base_algo.SN):
    def __init__(self, w_init, b_init):
        super().__init__(w_init, b_init)

    def fit(self, x_samples, y_samples, epochs=100, eta=0.01):
        self.clear(x_samples, y_samples)

        for i in range(epochs):
            dw, db = 0, 0
            for x, y in zip(x_samples, y_samples):
                dw += self.grad_w(x, y)
                db += self.grad_b(x, y)
            self.w -= eta * dw / x_samples.shape[0]
            self.b -= eta * db / x_samples.shape[0]
            self.append_log()
