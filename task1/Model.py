import numpy as np


def to_hotspot(y, n):
    new_y = np.zeros((y.size, n), dtype=int)
    new_y[np.arange(y.size), y] = 1
    return new_y


class Model:
    def __init__(self, kernel, distance, h=1):
        self.K = kernel
        self.d = distance
        self.h = h
    
    def fit(self, x, y, n_classes):
        self.x = x
        self.y_hot = to_hotspot(y, n_classes)
        self.gamma = np.zeros_like(y)
        cont = True
        while (cont):
            cont = False
            for i in range(self.x.shape[0]):
                if (self.predict_one(self.x[i]) != y[i]):
                    self.gamma[i] += 1
                    cont = True
        mask = self.gamma > 0
        self.gamma = self.gamma[mask]
        self.y_hot = self.y_hot[mask]
        self.x = self.x[mask]

    def predict(self, x):
        res = np.empty(x.shape[0], dtype=int)
        for i in range(res.size):
            res[i] = self.predict_one(x[i])
        return res

    def predict_one(self, x):
        rates = np.sum(self.y_hot * (self.gamma * self.K(self.d(x, self.x) / self.h)).reshape(-1, 1), 0)
        return np.argmax(rates)
    