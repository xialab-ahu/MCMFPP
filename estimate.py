import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_curve, auc, f1_score
from tqdm import tqdm
def Aiming(y_hat, y):
    """
    the “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    """

    n, m = y_hat.shape

    score_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:  # L ∪ L*
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:  # L ∩ L*
                intersection += 1
        if intersection == 0:
            continue
        score_k += intersection / sum(y_hat[v])
    return score_k / n


def Coverage(y_hat, y):
    """
    The “Coverage” rate (also called “Recall”) is to reflect the average ratio of the
    correctly predicted labels over the real labels; to measure the percentage of the
    real labels that are covered by the hits of prediction.
    """

    n, m = y_hat.shape

    score_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        score_k += intersection / sum(y[v])

    return score_k / n


def Accuracy(y_hat, y):
    """
    The “Accuracy” rate is to reflect the average ratio of correctly predicted labels
    over the total labels including correctly and incorrectly predicted labels as well
    as those real labels but are missed in the prediction
    """

    n, m = y_hat.shape

    score_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        score_k += intersection / union
    return score_k / n


def AbsoluteTrue(y_hat, y):
    """
    same
    """

    n, m = y_hat.shape
    score_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            score_k += 1
    return score_k / n

def AbsoluteFalse(y_hat, y):
    """
    hamming loss
    """

    n, m = y_hat.shape

    score_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        score_k += (union - intersection) / m
    return score_k / n


def evaluate(score_label, y, threshold1=0.6, threshold2=0.4):
    _, m = y.shape
    if isinstance(threshold1, (int, float)):
        threshold1 = [threshold1] * m
    if threshold1 is None:
        threshold1 = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    y_hat = np.copy(score_label)
    n=0
    for i in range(len(y_hat)):
        max=0
        k=0
        for j in range(len(y_hat[i])):
            if y_hat[i][j]>max:
                max=y_hat[i][j]
                k=j
            if y_hat[i][j] < threshold1[j]:  # threshold
                y_hat[i][j] = 0
            else:
                y_hat[i][j] = 1
        if y_hat[i].sum()==0:
            if max>threshold2:
                y_hat[i][k]=1
    aiming = Aiming(y_hat, y)
    coverage = Coverage(y_hat, y)
    accuracy = Accuracy(y_hat, y)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_false = AbsoluteFalse(y_hat, y)
    return aiming, coverage, accuracy, absolute_true, absolute_false

def calculate_true_and_predicted_positives(y_hat, y):
    true_positives = np.sum(y == 1, axis=0)
    predicted_positives = np.sum(y_hat, axis=0)
    TP = np.sum(np.logical_and(y == 1, y_hat == 1), axis=0)
    return true_positives, predicted_positives,TP

def calculate_true_and_predicted_negatives(y_hat, y):
    true_negatives = np.sum(1-y , axis=0)
    predicted_negatives = np.sum(1 - y_hat, axis=0)
    TN = np.sum(np.logical_and(y == 0, y_hat == 0), axis=0)
    return true_negatives, predicted_negatives,TN

def calculate_false_negatives(y_hat, y):
    false_negatives = np.sum(np.logical_and(y == 1, y_hat == 0), axis=0)
    return false_negatives

def calculate_false_positives(y_hat, y):
    false_positives = np.sum(np.logical_and(y == 0, y_hat == 1), axis=0)
    return false_positives


def obtain_functional_predictions(preds, threshold1=0.6, threshold2=0.4):
    y_hat = np.copy(preds)
    for i in range(len(y_hat)):
        max=0
        k=0
        for j in range(len(y_hat[i])):
            if y_hat[i][j]>max:
                max=y_hat[i][j]
                k=j
            if y_hat[i][j] < threshold1:
                y_hat[i][j] = 0
            else:
                y_hat[i][j] = 1
        if y_hat[i].sum()==0:
            if max>threshold2:
                y_hat[i][k]=1
    return y_hat