import csv
from typing import Dict, List

from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def compute_metrics(predict: List[float], y_true: List[int]) -> Dict[str, float]:
    auroc = roc_auc_score(y_true, predict)

    precision, recall, _ = precision_recall_curve(y_true, predict)
    auprc = auc(recall, precision)

    return {
        'auroc': auroc,
        'auprc': auprc
    }


def dump_metrics(metrics: Dict[str, float], path: str):
    with open(path, 'w') as f:
        w = csv.DictWriter(f, metrics.keys())
        w.writeheader()
        w.writerow(metrics)
