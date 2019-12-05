from . import base_algo


class MiniBatch(base_algo.SN):
    def __init__(self, w_init, b_init):
        super().__init__(w_init, b_init)

    def fit(self, x_samples, y_samples, epochs=100, eta=0.01, mini_batch_size=100):
        self.clear(x_samples, y_samples)

        for i in range(epochs):
            dw, db = 0, 0
            points_seen = 0
            for x, y in zip(x_samples, y_samples):
                dw += self.grad_w(x, y)
                db += self.grad_b(x, y)
                points_seen += 1
                if points_seen % mini_batch_size == 0:
                    self.w -= eta * dw / mini_batch_size
                    self.b -= eta * db / mini_batch_size
                    self.append_log()
                    dw, db = 0, 0
