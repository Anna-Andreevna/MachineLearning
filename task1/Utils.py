import numpy as np


def split_np(x, y, val_persentage, seed=42):
    np.random.seed(seed)
    
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    num_val = int(indices.size * val_persentage)

    train_indices = indices[:-num_val]
    train_x = x[train_indices]
    train_y = y[train_indices]

    val_indices = indices[-num_val:]
    val_x = x[val_indices]
    val_y = y[val_indices]

    return train_x, train_y, val_x, val_y


def split_pd(x, y, val_persentage, seed=42):
    np.random.seed(seed)
    
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    num_val = int(indices.size * val_persentage)

    train_mask = np.zeros_like(indices, dtype=bool)
    train_mask[indices[:-num_val]] = True
    train_x = x[train_mask]
    train_y = y[train_mask]

    val_mask = np.logical_not(train_mask)
    val_x = x[val_mask]
    val_y = y[val_mask]

    return train_x, train_y, val_x, val_y
