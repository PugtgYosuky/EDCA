from sklearn import metrics
import numpy as np

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