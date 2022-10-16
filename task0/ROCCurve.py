import numpy as np
import matplotlib.pyplot as plt

def TPR(preds, labels):
    if not np.any(labels):
        return 0
    return np.sum(np.logical_and(preds, labels)) / np.sum(labels)

def FPR(preds, labels):
    if np.all(labels):
        return 0
    return np.sum(np.logical_and(preds, np.logical_not(labels))) / np.sum(np.logical_not(labels))

class ROCCurve:
    xlabel = "FPR"
    ylabel = "TPR"
    title = "ROC Curve"
    
    def __init__(self, preds, labels):
        ind = np.argsort(preds)
        self.predictions = preds[ind]
        self.labels = labels[ind]

        self.fpr = np.empty(labels.size + 1, dtype=float)
        self.tpr = np.empty(labels.size + 1, dtype=float)

        eps = 1e-6
        for (i, pred) in enumerate(self.predictions):
            threshold = pred - eps
            self.fpr[-i-1] = FPR(self.predictions > threshold, self.labels)
            self.tpr[-i-1] = TPR(self.predictions > threshold, self.labels)
        self.fpr[0] = FPR(self.predictions > 1, self.labels)
        self.tpr[0] = TPR(self.predictions > 1, self.labels)
        
    def x(self):
        return self.fpr
    
    def y(self):
        return self.tpr
