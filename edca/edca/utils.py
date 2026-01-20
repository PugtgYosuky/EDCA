from sklearn import metrics
import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, equal_opportunity_difference
import pandas as pd

# for debug mode
import os
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"
def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print("[DEBUG]", *args, **kwargs)


def mcc_metric(y_true, y_pred, y_prob=None):
    """ Calculate the Matthews Correlation Coefficient normalized and inverted """
    mcc_normalized = (metrics.matthews_corrcoef(y_true, y_pred)+1)/2
    return 1 - mcc_normalized

def f1_metric(y_true, y_pred, y_prob=None):
    """ Calculate the F1 Score normalized and inverted """
    return 1 - metrics.f1_score(y_true, y_pred,  average='weighted')

def accuracy_metric(y_true, y_pred, y_prob=None):
    """ Calculate the Accuracy normalized and inverted """
    return 1 - metrics.accuracy_score(y_true, y_pred, normalize=True)

def precision_metric(y_true, y_pred, y_prob=None):
    """ Calculate the Precision normalized and inverted """
    return 1 - metrics.precision_score(y_true, y_pred, average='weighted')

def recall_metric(y_true, y_pred, y_prob=None):
    """ Calculate the Recall normalized and inverted """
    return 1 - metrics.recall_score(y_true, y_pred, average='weighted')

def mcc_recall_metric(y_true, y_pred, y_prob=None):
    """ Calculates the normalized weighted sum between MCC and Recall
    key : mcc_recall
    """
    return (mcc_metric(y_true, y_pred, y_prob)+recall_metric(y_true, y_pred, y_prob)) / 2

def roc_auc_metric(y_true, y_pred, y_prob=None):
    """ Calculate the ROC AUC normalized and inverted """
    if len(set(y_true)) == 2:
        # binary task
        return 1 - metrics.roc_auc_score(y_true, y_prob[:, 1])
    else:
        # multiclass task
        return 1 - metrics.roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')

def roc_precision_metric(y_true, y_pred, y_prob=None):
    """ Calculate the ROC AUC and the Precision combined and inverted """
    roc_auc = roc_auc_metric(y_true, y_pred, y_prob)
    precision = precision_metric(y_true, y_pred)
    return (roc_auc + precision) / 2

def mse_metric(y_true, y_pred, y_prob=None):
    """ Calculate the Mean Squared Error normalized"""
    y_mean = np.mean(y_true)
    mse_max = metrics.mean_squared_error(y_true, np.full_like(y_true, y_mean))
    mse = metrics.mean_squared_error(y_true, y_pred)
    return mse / mse_max

def rmse_metric(y_true, y_pred, y_prob=None):
    y_mean = np.mean(y_true)
    rmse_max = np.sqrt(metrics.mean_squared_error(y_true, np.full_like(y_true, y_mean)))
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    return rmse / rmse_max

def r2_metric(y_true, y_pred, y_prob=None):
    """ Calculate the R2 Score normalized """
    r2 = metrics.r2_score(y_true, y_pred)
    # all negative values are converted to 0
    r2 =  max(0, r2)
    # invert the metric so that the lower the better for a minimization problem
    return 1 - r2

def mape_metric(y_true, y_pred, y_prob=None):
    epsilon = 1e-10
    y_true_safe = np.maximum(np.abs(y_true), epsilon)
    return metrics.mean_absolute_percentage_error(y_true, y_pred)


def error_metric_function(metric_name, task):
    """ Method to select the metric to use on the EA based on it's name. All metrics are inverted to minimize the metric's error. All values are normalised between 0 and 1."""
    metric_name = metric_name.lower()
    if task == 'classification':
        if metric_name == 'mcc':
            return mcc_metric
        elif metric_name == 'f1' or metric_name == 'f-score':
            return f1_metric
        elif metric_name == 'accuracy':
            return accuracy_metric
        elif metric_name == 'precision':
            return precision_metric
        elif metric_name == 'recall':
            return recall_metric
        elif metric_name == 'roc_auc':
            return roc_auc_metric
        elif metric_name == 'roc_precision':
            return roc_precision_metric
        elif metric_name == 'mcc_recall':
            return mcc_recall_metric
        else:
            raise ValueError(f'Metric {metric_name} implemented yet.')
        
    elif task == 'regression':
        if metric_name == 'mae' or metric_name == 'mape':

            # always use the MAPE to be normalized
            return metrics.mean_absolute_percentage_error
        elif metric_name == 'mse':
            return mse_metric
        elif metric_name == 'rmse':
            return rmse_metric
        elif metric_name == 'r2':
            return r2_metric
        else:
            raise ValueError(f'Metric {metric_name} implemented yet.')
    else:
        raise ValueError(f'Task {task} not implemented yet.')
    

def class_distribution_distance(classes_proportions, number_classes):
    """
    Calculates the balance metric of the classes. 
    Mean Absolute Error between the proportion of each class and the ideal proportion of each class.
    
    Parameters:
    ----------
    classes_proportions : list
        List with the proportion of each class in the dataset
    
    number_classes : int
        Number of classes in the dataset
        
    Returns:
    -------
        float
            Balance metric
    """
    # Calculate the Class Distribution Distance (CDD)
    # ideal probability of each class
    ideal_prob = 1 / number_classes 
    # ensure all classes are present
    if len(classes_proportions) < number_classes:
        classes_proportions = np.append(classes_proportions, [0] * (number_classes - len(classes_proportions)))
    # calculate sum of absolute differences between the ideal probability and the actual probability of each class
    cdd = np.abs(classes_proportions - ideal_prob).sum()
    # worst case when all samples are all from 1 class
    upper_bound = (1-ideal_prob) + ideal_prob * (number_classes-1)
    return cdd/upper_bound


def has_full_coverage(X, sensitive_attributes, y_true):
    sensitive_data = X[sensitive_attributes].copy()
    sensitive_data['target'] = y_true
    stats = sensitive_data.groupby(sensitive_attributes)['target'].agg(['sum', 'count'])
    stats['negatives'] = stats['count'] - stats['sum']
    return ((stats['sum'] > 0) & (stats['negatives'] > 0)).all()

def get_tpr(y_true, y_proba, fpr_common):
    """
    Calculates the TPR for FPR
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_proba)
    # calculate TPR for a common FPR
    return np.interp(fpr_common, fpr, tpr)


def abroca_metric(sensitive_data, y_true, y_proba, points=100):
    """
    Calculates the Absolute Between ROC Area (ABROCA) based on (https://doi.org/10.1145/3303772.3303791) metric for fairness. 
    It allows for multiple sensitive features, not just pairwise comparisons since it compares the best and the worst curves based on their AUCs 

    Params:
        sensitive_data: pd.DataFrame 
            Dataframe with the sensitive attributes. Can have more than two values
        y_true : pd.Series or np.array or list
            Real values of the target. Should be in true or false / 0 or 1
        y_proba : pd.Series or np.array or list
            Probability of being the positive class for each example
    """
    df = pd.DataFrame(sensitive_data)
    if df.shape[1] > 1:
        # aggregate the sensitive features into just one
        combined_sensitive = df.astype(str).agg('_'.join, axis=1)
        combined_sensitive.name = 'sensitives'
        df = pd.DataFrame(combined_sensitive)
        cols = 'sensitives'
    else:
        cols = list(df.columns)
    df['y_true'] = list(y_true)
    df['y_proba'] = list(y_proba)
    # common FPR
    fpr_common = np.linspace(0, 1, points)
    # calculate TPR for all the groups of sensitive attributes
    tprs = df.groupby(by=cols).apply(lambda x: get_tpr(x['y_true'], x['y_proba'], fpr_common)).values
    # calculate best and worst curves
    rocs = [metrics.auc(tpr, fpr_common) for tpr in tprs]
    best_curve = tprs[np.nanargmax(rocs)]
    worst_curve = tprs[np.nanargmin(rocs)]
    # calculate area
    return np.trapezoid(np.abs(best_curve - worst_curve), fpr_common)


def fairness_metric(y_true, y_pred, y_proba, X_test, fairness_params):
    """ calculates the fairness value based on the selected fairness metric. It could be:
        - demographic_parity
        - equal_opportunity
        - equalized_odds
    The fairness params receive the weights for each metric. In the end, it outputs the weighted sum of the fairness metrics.
    The output value is normalized between 0 and 1.
    """
    output_fairness = 0
    # calculate each fairness metric
    y_true_transformed = pd.Series(y_true) == fairness_params['positive_class']
    y_pred_transformed = pd.Series(y_pred) == fairness_params['positive_class']
    combined_sensitive = X_test[fairness_params['sensitive_attributes']].astype(str).agg('_'.join, axis=1)
    combined_sensitive.name = 'sensitives'
    
    fair_metrics = {
        'demographic_parity' : demographic_parity_difference(y_true_transformed, y_pred_transformed, sensitive_features=combined_sensitive,),
        'equal_opportunity' : equal_opportunity_difference(y_true_transformed, y_pred_transformed, sensitive_features=combined_sensitive,),
        'equalized_odds' : equalized_odds_difference(y_true_transformed, y_pred_transformed, sensitive_features=combined_sensitive,),
        'abroca' : abroca_metric(sensitive_data=combined_sensitive, y_true=y_true_transformed, y_proba=y_proba)
    }
    # calculate weighted sum of fairness metrics
    if 'demographic_parity' in fairness_params:
        output_fairness += (fairness_params['demographic_parity'] * abs(fair_metrics['demographic_parity']))
    if 'equal_opportunity' in fairness_params:
        output_fairness += (fairness_params['equal_opportunity'] * abs(fair_metrics['equal_opportunity']))
    if 'equalized_odds' in fairness_params:
        output_fairness += (fairness_params['equalized_odds'] * abs(fair_metrics['equalized_odds']))
    if 'abroca' in fairness_params:
        output_fairness += (fairness_params['abroca'] * abs(fair_metrics['abroca']))
    fair_metrics['fairness_metric'] = output_fairness
    
    # ensure an value or 1=error if the end fairness metric is 0 for when it does not have full coverage
    if output_fairness == 0.0 and (not fairness_params['full_coverage']):
        return 1, fair_metrics
    return output_fairness, fair_metrics