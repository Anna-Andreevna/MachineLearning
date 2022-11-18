import numpy as np
import matplotlib.pyplot as plt

def Precision(preds, labels):
    return np.sum(np.logical_and(preds, labels), 0) / (np.sum(preds, 0) + 1e-8)

def Recall(preds, labels):
    return np.sum(np.logical_and(preds, labels), 0) / (np.sum(labels, 0) + 1e-8)

class PRCurve:
    xlabel = "Recall"
    ylabel = "Precision"
    title = "PR Curve"
    
    def __init__(self, preds, labels):
        ind = np.argsort(preds)
        self.predictions = preds[ind]
        self.labels = labels[ind]

        self.precisions = np.empty(labels.size + 1, dtype=float)
        self.recalls = np.empty(labels.size + 1, dtype=float)

        threshold = np.flip(self.predictions).reshape(1, -1) - 1e-6
        self.precisions[1:] = self.Precision(threshold)
        self.recalls[1:] = self.Recall(threshold)
        self.precisions[0] = self.Precision(1)
        self.recalls[0] = self.Recall(1)
        
    def Precision(self, threshold):
        p = Precision(self.predictions.reshape(-1, 1) > threshold, self.labels.reshape(-1, 1))
        return p if p.size > 1 else p[0]
    
    def Recall(self, threshold):
        r = Recall(self.predictions.reshape(-1, 1) > threshold, self.labels.reshape(-1, 1))
        return r if r.size > 1 else r[0]
    
    def x(self):
        return self.recalls
    
    def y(self):
        return self.precisions
