import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import pairwise_kernels
from sklearn.metrics import confusion_matrix

def chi_squared_distance(x, y):
    eps = 1e-10
    return np.sum((x - y) **2 / (x + y + eps))

def show_confusion_matrix(y_test, y_pred):
    labels = list(dict.fromkeys(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def spatial_pyramid_kernel_creator(levels):
    def spatial_pyramid_kernel(hist1, hist2):
        total_similarity = 0
        for level in range(levels + 1):
            weight = 1 / (2**(levels - level))
            similarity = 0
            num_cells = 2**level
            cell_size = len(hist1) // (num_cells**2)
            for cell in range(num_cells**2):
                start = cell * cell_size
                end = start + cell_size
                similarity += np.sum(np.minimum(hist1[start:end], hist2[start:end]))
            total_similarity += weight * similarity
        return total_similarity
    return spatial_pyramid_kernel

def compute_kernel_matrix(X_train, X_test, levels):
    K_train = pairwise_kernels(X_train, X_train, metric=spatial_pyramid_kernel_creator(levels))
    K_test = pairwise_kernels(X_test, X_train, metric=spatial_pyramid_kernel_creator(levels))
    return K_train, K_test