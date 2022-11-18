import numpy as np
import matplotlib.pyplot as plt

def TPR(preds, labels):
    return np.sum(np.logical_and(preds, labels), 0) / (np.sum(labels, 0) + 1e-8)

def FPR(preds, labels):
    return np.sum(np.logical_and(preds, np.logical_not(labels)), 0) / (np.sum(np.logical_not(labels), 0) + 1e-8)

class ROCCurve:
    xlabel = "FPR"
    ylabel = "TPR"
    title = "ROC Curve"
    
    def __init__(self, preds, labels, thr_preds=None):
        ind = np.argsort(preds)
        self.predictions = preds[ind]
        self.labels = labels[ind]

        threshold = np.flip(self.predictions if thr_preds is None else thr_preds).reshape(1, -1) - 1e-6
        self.fpr = np.empty(threshold.size + 1, dtype=float)
        self.tpr = np.empty_like(self.fpr)

        
        self.fpr[1:] = self.FPR(threshold)
        self.tpr[1:] = self.TPR(threshold)
        self.fpr[0] = self.FPR(1)
        self.tpr[0] = self.TPR(1)
        
    def TPR(self, threshold):
        t = TPR(self.predictions.reshape(-1, 1) > threshold, self.labels.reshape(-1, 1))
        return t if t.size > 1 else t[0]
    
    def FPR(self, threshold):
        f = FPR(self.predictions.reshape(-1, 1) > threshold, self.labels.reshape(-1, 1))
        return f if f.size > 1 else f[0]
    
    def x(self):
        return self.fpr
    
    def y(self):
        return self.tpr
    
class MicroROCCurve:
    xlabel = "FPR"
    ylabel = "TPR"
    title = "MicroROC Curve"
    
    def __init__(self, preds, labels):
        thr_preds = np.sort(preds.reshape(-1))
        fpr = np.empty((thr_preds.size+1, preds.shape[1]))
        tpr = np.empty_like(fpr)
        for i in range(preds.shape[1]):
            roc = ROCCurve(preds[:, i], labels[:, i], thr_preds)
            fpr[:, i] = roc.fpr
            tpr[:, i] = roc.tpr
        self.fpr = np.mean(fpr, 1)
        self.tpr = np.mean(tpr, 1)

    def x(self):
        return self.fpr
    
    def y(self):
        return self.tpr
    
class MacroROCCurve:
    xlabel = "FPR"
    ylabel = "TPR"
    title = "MacroROC Curve"
    
    def __init__(self, preds, labels):
        roc = ROCCurve(preds.reshape(-1), labels.reshape(-1))
        self.fpr = roc.fpr
        self.tpr = roc.tpr

    def x(self):
        return self.fpr
    
    def y(self):
        return self.tpr
    
