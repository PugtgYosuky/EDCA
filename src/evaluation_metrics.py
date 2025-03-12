from sklearn import metrics
import time

def evo_mcc_metric(y_test, y_pred):
    """MCC metric for the evolutionary algorithm"""
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    return 1 - mcc


def evo_f1_metric(y_test, y_pred):
    """F1 metric for the evolutionary algorithm"""
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    return 1 - f1


def evo_metric(metric_name):
    """Method to select the metric to use on the evolutionary algorithm based on metric's name"""
    if metric_name == 'mcc':
        return evo_mcc_metric
    elif metric_name == 'f1':
        return evo_f1_metric
    else:
        raise ValueError('No metric with that Name')


def flaml_f1_metric(
        X_val,
        y_val,
        estimator,
        labels,
        X_train,
        y_train,
        weight_val=None,
        weight_train=None,
        *args):
    """F1 metric for the flaml framework"""
    start = time.time()
    y_pred = estimator.predict(X_val)
    end = time.time()
    pred_time = (end - start)
    f1 = metrics.f1_score(y_val, y_pred, average='weighted')
    inv_f1 = 1 - f1
    return inv_f1, {
        'inv_f1': inv_f1,
        'f1': f1,
        'pred_time': pred_time
    }


def flaml_mcc_metric(
        X_val,
        y_val,
        estimator,
        labels,
        X_train,
        y_train,
        weight_val=None,
        weight_train=None,
        *args):
    """MCC metric for the flaml framework"""
    start = time.time()
    y_pred = estimator.predict(X_val)
    end = time.time()
    pred_time = (end - start)
    mcc = metrics.matthews_corrcoef(y_val, y_pred)
    inv_mcc = 1 - mcc
    return inv_mcc, {
        'inv_mcc': inv_mcc,
        'mcc': mcc,
        'pred_time': pred_time
    }


def flaml_metric(metric_name):
    """Method to select the metric for the flaml framework"""
    if metric_name == 'mcc':
        return flaml_mcc_metric
    elif metric_name == 'f1':
        return 'f1'
    elif metric_name == 'roc_auc':
        return 'roc_auc'
    raise ValueError('No metric with that name')


def tpot_mcc_metric(y_test, y_pred):
    """MCC metric for the evolutionary algorithm"""
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    return 1 - mcc


def tpot_f1_metric(y_test, y_pred):
    """F1 metric for the evolutionary algorithm"""
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    return 1 - f1


def tpot_metric(metric_name):
    """Method to select the metric to use on the evolutionary algorithm based on metric's name"""
    if metric_name == 'mcc':
        return tpot_mcc_metric
    elif metric_name == 'f1':
        return tpot_f1_metric
    else:
        raise ValueError('No metric with that Name')