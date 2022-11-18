import numpy as np
import time


def to_hotspot(y, n):
    new_y = np.zeros((y.size, n), dtype=int)
    new_y[np.arange(y.size), y] = 1
    return new_y

def accuracy(y, gt):
    return np.sum(y == gt) / y.size

class Model:
    def __init__(self, kernel, distance, h=1):
        self.K = kernel
        self.d = distance
        self.h = h

    def fit(self, x, y, n_classes, max_time=None, min_train_acc=1) -> bool:
        self.x = x
        self.y_hot = to_hotspot(y, n_classes)
        self.gamma = np.full(y.shape, 0)
        cont = True
        t = time.time()
        while (accuracy(self.predict(self.x), y) < min_train_acc):
            if max_time is not None and time.time() - t >= max_time:
                return False
            for i in range(self.x.shape[0]):
                if (self.predict_one(self.x[i]) != y[i]):
                    self.gamma[i] += 1
        mask = self.gamma > 0
        self.gamma = self.gamma[mask]
        self.y_hot = self.y_hot[mask]
        self.x = self.x[mask]
        self.y = y[mask]
        return True
        

    def predict(self, x):
        # (n_x, n_selfX, n_class)
        y_hot = np.expand_dims(self.y_hot, 0) # duplicate for each x_i
        gamma = np.expand_dims(self.gamma, (0,2)) # duplicate for each x_i and each class
        test_x = np.expand_dims(x, 1) # duplicate for each self_x_i
        train_x = np.expand_dims(self.x, 0) # duplicate for each x_i
        distances = np.expand_dims(self.K(self.d(test_x, train_x) / self.h), 2) # duplicate for each class
        rates = np.sum(y_hot * gamma * distances, axis=1)
        res = np.argmax(rates, axis=1)
        return res

    def predict_one(self, x):
        rates = np.sum(self.y_hot * (self.gamma * self.K(self.d(x, self.x) / self.h)).reshape(-1, 1), 0)
        return np.argmax(rates)
    