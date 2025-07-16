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


def evaluate_with_f1(score_label, y, threshold1=0.6, threshold2=0.4):
    """
    Enhanced evaluation function that includes micro-F1 and macro-F1 metrics
    in addition to the original 5 metrics.

    Returns:
        List of 7 metrics: [aiming, coverage, accuracy, absolute_true, absolute_false, micro_f1, macro_f1]
    """
    _, m = y.shape
    if isinstance(threshold1, (int, float)):
        threshold1 = [threshold1] * m
    if threshold1 is None:
        threshold1 = [0.8, 0.5, 0.7, 0.6, 0.3, 0.3, 0.5, 0.7, 0.8, 0.7, 0.2, 0.9, 0.7, 0.8, 0.5, 0.7, 0.6, 0.8, 0.7,
                      0.7, 0.9]
    y_hat = np.copy(score_label)
    for i in range(len(y_hat)):
        max_val = 0
        max_idx = 0
        for j in range(len(y_hat[i])):
            if y_hat[i][j] > max_val:
                max_val = y_hat[i][j]
                max_idx = j
            if y_hat[i][j] < threshold1[j]:
                y_hat[i][j] = 0
            else:
                y_hat[i][j] = 1
        # 当序列所有功能预测概率都低于threshold1时，预测一个最有可能的功能，如果其概率高于threshold2
        if y_hat[i].sum() == 0:
            if max_val > threshold2:
                y_hat[i][max_idx] = 1

    # 计算原有的5个指标
    aiming = Aiming(y_hat, y)
    coverage = Coverage(y_hat, y)
    accuracy = Accuracy(y_hat, y)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_false = AbsoluteFalse(y_hat, y)

    # 计算新增的micro-F1和macro-F1指标
    micro_f1 = MicroF1(y, y_hat)
    macro_f1 = Macro_F1(y, y_hat)

    return aiming, coverage, accuracy, absolute_true, absolute_false, micro_f1, macro_f1


def MicroF1(y_true, y_pred):
    """
    Calculate micro-averaged F1 score
    """
    # Calculate global TP, FP, FN
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))

    # Calculate micro precision and recall
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    # Calculate micro F1
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1


def Macro_F1(y_true, y_pred):
    """
    Calculate macro-averaged F1 score
    """
    n_labels = y_true.shape[1]
    f1_scores = []

    for i in range(n_labels):
        # Calculate TP, FP, FN for each label
        tp = np.sum(np.logical_and(y_true[:, i] == 1, y_pred[:, i] == 1))
        fp = np.sum(np.logical_and(y_true[:, i] == 0, y_pred[:, i] == 1))
        fn = np.sum(np.logical_and(y_true[:, i] == 1, y_pred[:, i] == 0))

        # Calculate precision and recall for this label
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        # Calculate F1 for this label
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1_scores.append(f1)

    # Return macro-averaged F1
    return np.mean(f1_scores)