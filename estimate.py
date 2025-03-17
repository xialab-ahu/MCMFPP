import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_curve, auc, f1_score
from tqdm import tqdm
def compute_metric_labelwise(labels, outputs, thres, show_detail = True):#这里thres值为0.6
    results_all = []
    _, m = labels.shape
    if isinstance(thres, (int, float)):
        thres = [thres] * m
    for i in range(m):
        labels_i, outputs_i = labels[:, i], outputs[:, i]
        tn, fp, fn, tp = confusion_matrix(labels_i, outputs_i > thres[i]).ravel()
        accuracy = (tp + tn) / (tn + fp + fn + tp + 1e-20)
        sensitivity = tp / (tp + fp + 1e-20)
        specificity = tn / (tn + fp + 1e-20)
        precision = sensitivity
        recall = tp / (tp+fn + + 1e-20)
        f1score = 2 * (precision * recall) / (precision + recall + 1e-20)
        mcc = matthews_corrcoef(labels_i, outputs_i > thres[i])
        auc_val, precision, recall = compute_auroc(labels_i, outputs_i)
        results_all.append([auc_val, accuracy, sensitivity, specificity, f1score, mcc])
        # if show_detail:
        #     print(f'{round(auc_val, 3)}\t{round(accuracy,3)}\t{round(sensitivity,3)}\t{round(specificity,3)}\t{round(f1score,3)}\t{round(mcc, 3)}')
    results_all = np.array(results_all)
    metric_all = np.mean(results_all, axis = 0)
    auc_val, accuracy, sensitivity, specificity, f1score, mcc = metric_all
    # print('Average')
    # print(f'{round(auc_val, 3)}\t{round(accuracy,3)}\t{round(sensitivity,3)}\t{round(specificity,3)}\t{round(f1score,3)}\t{round(mcc, 3)}')
    return round(auc_val, 3),round(accuracy,3),round(sensitivity,3),round(specificity,3),round(f1score,3),round(mcc, 3)#添加的代码


def compute_auroc(labels, outputs):
    fpr, tpr, _ = roc_curve(labels, outputs)
    auc_val = auc(fpr, tpr)
    return auc_val, fpr, tpr



# 多标签评价指标

def Aiming(y_hat, y):#预测正确的正例数据占预测为正例数据的比例。
    """
    the “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    """

    n, m = y_hat.shape  # n表示肽的数量， m表示肽的功能数量

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


def Coverage(y_hat, y):#反映了分类器识别出的正样本占所有实际正样本的比例。即召回率是指对于所有真实正类样本
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


def Accuracy(y_hat, y):#标签预测的正确率
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
            # union += 1#添加代码
            # if y_hat[v, h] == y[v, h]:添加代码
            #     intersection+=1添加代码
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






# def evaluate(score_label, y, threshold=0.6):
#     # 将预测概率分数转为标签
#     _, m = y.shape
#     if isinstance(threshold, (int, float)):
#         threshold = [threshold] * m
#     if threshold is None:
#         threshold = [0.8, 0.5, 0.7, 0.6, 0.3, 0.3, 0.5, 0.7, 0.8, 0.7, 0.2, 0.9, 0.7, 0.8, 0.5, 0.7, 0.6, 0.8, 0.7, 0.7, 0.9]
#     auc_val,binary_accuracy,sensitivity,specificity,f1score,mcc=compute_metric_labelwise(y,score_label,threshold)
#     y_hat = np.copy(score_label)
#     for i in range(len(y_hat)):
#         for j in range(len(y_hat[i])):
#             if y_hat[i][j] < threshold[j]:  # threshold
#                 y_hat[i][j] = 0
#             else:
#                 y_hat[i][j] = 1
#
#     # MCC = []
#     # from sklearn.metrics import matthews_corrcoef
#     #
#     # for k in range(21):
#     #     M = matthews_corrcoef(y[:, k], y_hat[:, k])
#     #     MCC.append(M)
#
#     # 评估模型
#     aiming = Aiming(y_hat, y)
#     coverage = Coverage(y_hat, y)
#     accuracy = Accuracy(y_hat, y)
#     absolute_true = AbsoluteTrue(y_hat, y)
#     absolute_false = AbsoluteFalse(y_hat, y)
#
#     # return aiming, coverage, accuracy, absolute_true, absolute_false#源代码
#     return aiming, coverage, accuracy, absolute_true, absolute_false,auc_val,binary_accuracy,sensitivity,specificity,f1score,mcc#添加代码



def evaluate(score_label, y, threshold=0.6):
    # 将预测概率分数转为标签
    _, m = y.shape
    if isinstance(threshold, (int, float)):
        threshold = [threshold] * m
    if threshold is None:
        threshold = [0.8, 0.5, 0.7, 0.6, 0.3, 0.3, 0.5, 0.7, 0.8, 0.7, 0.2, 0.9, 0.7, 0.8, 0.5, 0.7, 0.6, 0.8, 0.7, 0.7, 0.9]
    # auc_val,binary_accuracy,sensitivity,specificity,f1score,mcc=compute_metric_labelwise(y,score_label,threshold)
    # class_level_map = label_mAP(y, score_label)
    # sample_level_map = sample_mAP(y, score_label)
    y_hat = np.copy(score_label)
    n=0
    for i in range(len(y_hat)):
        max=0
        k=0
        for j in range(len(y_hat[i])):
            if y_hat[i][j]>max:
                max=y_hat[i][j]
                k=j
            if y_hat[i][j] < threshold[j]:  # threshold
                y_hat[i][j] = 0
            else:
                y_hat[i][j] = 1
        if y_hat[i].sum()==0:#当序列所有功能预测概率都低于0.6时，预测一个最有可能的功能，如果其概率高于0.4
            if max>0.4:
                y_hat[i][k]=1

    # 评估模型
    aiming = Aiming(y_hat, y)
    coverage = Coverage(y_hat, y)
    accuracy = Accuracy(y_hat, y)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_false = AbsoluteFalse(y_hat, y)
    # microF1 = MicroF1(y, y_hat)
    # macro_F1 = Macro_F1(y, y_hat)

    # return aiming, coverage, accuracy, absolute_true, absolute_false, class_level_map, sample_level_map, microF1, macro_F1#源代码
    return aiming, coverage, accuracy, absolute_true, absolute_false
    # return (aiming, coverage, accuracy, absolute_true, absolute_false,auc_val,binary_accuracy,sensitivity,specificity,f1score,mcc)#添加代码



# 多标签评价指标

# import numpy as np
# from torchmetrics.classification import MultilabelPrecision
# from libauc.metrics import auc_roc_score
#
#
# def metrics(pred, label):
#     # "Robust Asymmetric Loss for Multi-Label Long-Tailed Learning" CVPR-23
#     # 衡量长尾分布显着性的不平衡比记为Nmax/Nmin，其中N为每类中每类样本的数量。
#     # https://github.com/kalelpark/RALoss
#     # 项目路径：F:\Work\work3\ReCode\RAL-main
#     """mean Average Precision(mAP), mean Area Under Curve(mAUC), and F1-Score
#     considering the multi-label dataset."""
#     metric_precision = MultilabelPrecision(num_labels=26, average="macro")
#     precision = metric_precision(pred, label)
#     auc_score = np.mean(auc_roc_score(label, pred))
#
#     pred = pred > 0.5
#     acc = (pred == label).float().mean()
#
#     return precision, acc, auc_score


def prob_labels(score_label, y, threshold=0.5):
    # 将预测概率分数转为标签
    # 添加代码
    _, m = y.shape
    if isinstance(threshold, (int, float)):
        threshold = [threshold] * m
    if threshold is None:
        threshold = [0.8, 0.5, 0.7, 0.6, 0.3, 0.3, 0.5, 0.7, 0.8, 0.7, 0.2, 0.9, 0.7, 0.8, 0.5, 0.7, 0.6, 0.8, 0.7, 0.7, 0.9]
    #添加代码结束
    y_hat =  np.copy(score_label)
    for i in range(len(y_hat)):
        for j in range(len(y_hat[i])):
            if y_hat[i][j] < threshold[j]:
                y_hat[i][j] = 0
            else:
                y_hat[i][j] = 1
    return y_hat
def calculate_true_and_predicted_positives(y_hat, y):
    true_positives = np.sum(y == 1, axis=0)#TP
    predicted_positives = np.sum(y_hat, axis=0)#预测正确的数量
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
def plot_comparison_bar_chart(true_values, predicted_values, class_names, title, ylabel):
    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, true_values, width, label='True')
    rects2 = ax.bar(x + width/2, predicted_values, width, label='Predicted')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()




# def calculate_map(y_true, y_scores):
#     # 计算每个标签的平均精度
#     ap_list = []
#     for i in range(y_true.shape[1]):
#         ap = average_precision_score(y_true[:, i], y_scores[:, i])
#         ap_list.append(ap)
#     # 计算 mAP
#     map_score = np.mean(ap_list)
#     return map_score, ap_list
#
# def plot_map(ap_list, class_names, title='Average Precision per Class'):
#     x = np.arange(len(class_names))
#     fig, ax = plt.subplots(figsize=(12, 8))
#     rects = ax.bar(x, ap_list, label='Average Precision')
#
#     ax.set_ylabel('Average Precision')
#     ax.set_title(title)
#     ax.set_xticks(x)
#     ax.set_xticklabels(class_names, rotation=45, ha='right')
#     ax.legend()
#
#     plt.tight_layout()
#     plt.show()

#遍历找到最适合阈值
def find_optimal_threshold(score_label, y, class_index, min_threshold=0.960, max_threshold=0.980, step=0.00001):
    best_threshold = 0
    min_sum_fn_fp = 10000

    for threshold1 in tqdm(np.arange(min_threshold, max_threshold, step)):
        y_hat1 = prob_labels(score_label, y, threshold1)
        false_negatives = calculate_false_negatives(y_hat1, y)
        false_positives = calculate_false_positives(y_hat1, y)
        sum_fn_fp = false_negatives[class_index] + false_positives[class_index]

        if sum_fn_fp <min_sum_fn_fp:
            min_sum_fn_fp = sum_fn_fp
            best_threshold = threshold1

    return best_threshold, min_sum_fn_fp


# 定义AP函数
def AP(output, target):
    if len(target) == 0 or np.all(target == 0):
        return -1

    epsilon = 1e-8
    indices = output.argsort()[::-1]
    sorted_targets = target[indices]
    cumulative_hits = np.cumsum(sorted_targets)
    precision_at_i = cumulative_hits / (np.arange(len(output)) + 1)
    ap = np.sum(precision_at_i * sorted_targets) / (np.sum(sorted_targets) + epsilon)

    return ap

def label_mAP(targs, preds):
    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    for k in range(preds.shape[1]):
        scores = preds[:, k]
        targets = targs[:, k]
        ap[k] = AP(scores, targets)
    return 100 * ap.mean()

def sample_mAP(targs, preds):
    sample_ap = np.zeros(targs.shape[0])
    for i in range(targs.shape[0]):
        sample_ap[i] = AP(preds[i], targs[i])
    return 100 * sample_ap.mean()


def MicroF1(y_true, y_pred):
    r"""
	微平均F1（Micro-F1）
	表达式：
		\begin{equation}Micro-F1(h)=\frac{2\sum_{i=1}^{m}|Y_i\cap h(x_i)|}{\sum_{i=1}^{m}|Y_i|+\sum_{i=1}^{m}|h(x_i)|}\end{equation}
		其中，m表示样本总数，q表示标签总数，h(x_{i})表示样本i的预测标签，Y_{i}表示样本i的真实标签。\cap 表示集合的交集，
		\sum_{i=1}^{m}表示对m个样本求和。

	描述：
			微平均F1是精准率和召回率的调和平均数，它考虑了所有标签的预测结果。它计算的是正确预测的标签的数量除以实际为正的标签数量和预测为正的标签数量之和。

	参考文献：
		[1]. Zhang M-L, Zhou Z-H. A Review on Multi-Label Learning Algorithms.
		IEEE Transactions on Knowledge and Data Engineering 2014; 26(8): 1819–1837.

	参数：
		y_true(ndarray)：真实标签，shape=(m, n_labels)  其中m表示样本总数，n_labels表示标签总数，要求array元素为0或1
		y_pred(ndarray)：预测标签，shape=(m, n_labels)  其中m表示样本总数，n_labels表示标签总数，要求array元素为0或1

	返回：
		微平均F1值，shape=1
	"""
    return f1_score(y_true, y_pred, average='micro')

def Macro_F1(y_true, y_pred):
    """

	计算 Macro F1 指标。

	参数:
		- y_true(ndarray)：真实标签的集合，一个形状为 (m, n_labels) 的二维数组。
			  每个元素为0或1，表示实例是否具有某个标签。
		- y_pred(ndarray): 预测标签的集合，一个形状为 (m, n_labels) 的二维实数矩阵。
			  每个元素为0或1，表示实例是否具有某个标签。
	参考文献:
			[1]. Zhang M-L, Zhou Z-H. A Review on Multi-Label Learning Algorithms
				IEEE Transactions on Knowledge and Data Engineering 2014; 26(8): 1819–1837.
	返回：
			- f1_score_value: 计算出的 Macro F1 指标的值。其值越大越好。
	"""
    return f1_score(y_true, y_pred, average='macro')