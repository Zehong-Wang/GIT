import numpy as np
import torch.nn.functional as F
import torch
from torchmetrics import Accuracy, AUROC
from sklearn.metrics import f1_score, roc_auc_score

task2metric = {'node': 'acc', 'edge': 'acc', 'graph': 'auc', 'link_pred': 'auc'}


def evaluate(pred, y, mask=None, params=None):
    metric = task2metric[params['task']]

    if metric == 'acc':
        return eval_acc(pred, y, mask) * 100
    elif metric == 'auc':
        return eval_auc(pred, y) * 100
    else:
        raise ValueError(f"Metric {metric} is not supported.")


def eval_acc(y_pred, y_true, mask):
    device = y_pred.device
    num_classes = y_pred.size(1)

    evaluator = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    if mask is not None:
        return evaluator(y_pred[mask], y_true[mask]).item()
    else:
        return evaluator(y_pred, y_true).item()


# def eval_auc(y_pred, y_true):
#     y_pred = y_pred.view(-1)
#     y_true = y_true.view(-1)
#
#     mask = ~torch.isnan(y_true)
#
#     y_pred = y_pred[mask]
#     y_true = y_true[mask]
#
#     evaluator = AUROC(task='binary').to(y_pred.device)
#
#     return evaluator(y_pred, y_true).item()


def eval_auc(y_pred, y_true):
    ndim = y_true.ndim
    if ndim == 1:
        y_pred = y_pred.view(-1, 1)
        y_true = y_true.view(-1, 1)
    elif ndim == 2:
        pass

    rocauc_list = []
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)

# def eval_auc(y_scores, y_true):
#     y_true[y_true == 0] = -1
#
#     y_scores = y_scores.detach().cpu().numpy()
#     y_true = y_true.detach().cpu().numpy()
#
#     roc_list = []
#     for i in range(y_true.shape[1]):
#         # AUC is only defined when there is at least one positive data.
#         if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
#             is_valid = y_true[:, i] ** 2 > 0
#             roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
#
#     if len(roc_list) < y_true.shape[1]:
#         print("Some target is missing!")
#         print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))
#
#     return sum(roc_list) / len(roc_list)  # y_true.shape[1]

# def evaluate_based_on_metric(y_pred, y_true, metric):
#     if metric == "acc":
#         return eval_acc(y_pred, y_true)
#     elif metric == "auc":
#         y_true = y_true.unsqueeze(1)
#         return eval_rocauc(y_pred, y_true)
#     elif metric == "f1":
#         return eval_f1(y_pred, y_true)
#     else:
#         raise NotImplementedError("The metric is not supported!")
#
#
# def eval_rocauc(y_pred, y_true):
#     """adapted from ogb
#     https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
#     rocauc_list = []
#     y_true = y_true.detach().cpu().numpy()
#     if y_true.shape[1] == 1:
#         # use the predicted class for single-class classification
#         y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).detach().cpu().numpy()
#     else:
#         y_pred = y_pred.detach().cpu().numpy()
#
#     for i in range(y_true.shape[1]):
#         # AUC is only defined when there is at least one positive data.
#         if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
#             is_labeled = y_true[:, i] == y_true[:, i]
#             score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
#
#             rocauc_list.append(score)
#
#     if len(rocauc_list) == 0:
#         raise RuntimeError(
#             "No positively labeled data available. Cannot compute ROC-AUC."
#         )
#
#     return sum(rocauc_list) / len(rocauc_list)
#
#
# def eval_f1(y_pred, y_true):
#     y_true = y_true.detach().cpu().numpy()
#     y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
#     f1 = f1_score(y_true, y_pred, average="weighted")
#     # macro_f1 = f1_score(y_true, y_pred, average='macro')
#     return f1
