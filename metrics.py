import torch

import numpy as np
from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = np.array(y_true)

    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = np.array(y_pred)

    y_true = y_true.reshape(-1)
    y_pred = np.argmax(y_pred, axis=1).reshape(-1)

    res = confusion_matrix(y_true, y_pred)

    return res