import numpy as np
import matplotlib.pyplot as plt

def Precision(preds, labels):
    if not np.any(preds):
        return 0
    return np.sum(np.logical_and(preds, labels)) / np.sum(preds)

def Recall(preds, labels):
    if not np.any(labels):
        return 0
    return np.sum(np.logical_and(preds, labels)) / np.sum(labels)

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

        eps = 1e-6
        for (i, pred) in enumerate(self.predictions):
            threshold = pred - eps
            self.precisions[-i-1] = Precision(self.predictions > threshold, self.labels)
            self.recalls[-i-1] = Recall(self.predictions > threshold, self.labels)
        self.precisions[0] = Precision(self.predictions > 1, self.labels)
        self.recalls[0] = Recall(self.predictions > 1, self.labels)
        
    def x(self):
        return self.recalls
    
    def y(self):
        return self.precisions
