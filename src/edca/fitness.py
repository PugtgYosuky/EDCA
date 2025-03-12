import time
import numpy as np
import json
from edca.estimator import PipelineEstimator
from edca.utils import class_distribution_distance

from sklearn import set_config
set_config(transform_output='pandas')


def individual_fitness(
        X_train,
        X_val,
        y_train,
        y_val,
        pipeline_config,
        metric,
        individual, 
        individual_id):
    """
    Calculates the fitness of a given individual

    It instantiates the individual / pipeline based on its configurations and evaluates the predictions
    made.

    Parameters:
    ----------
    X_train : pandas.DataFrame
        Data to train the individual

    X_val: pandas.DataFrame
        Validation data

    y_train: pandas.Series
        Train target

    y_val : pandas.Series
        Validation target (to evaluate the predictions)

    pipeline_config : dict
        Data characteristics by type and components of the fitness (weights)

    metric : function
        Metric used to evaluate the predictions

    individual : dict
        Configuration of the individual

    Returns:
    -------
        tuple
            (float, float, float, float)
            Fitness, Metric result, Train Percentage, CPU time
    """
    # make a copy
    try:
        pipeline_estimator = PipelineEstimator(
            individual_config=individual,
            pipeline_config=pipeline_config,
            individual_id=individual_id
        )
        # start counter
        init_time = time.time()
        # train and predict values
        pipeline_estimator.fit(X_train, y_train)
        preds = pipeline_estimator.predict(X_val)
        preds_proba = pipeline_estimator.predict_proba(X_val)
        # end counter
        end_time = time.time()

        # calculate metrics
        pred_metric = metric(y_true=y_val, y_pred=preds, y_prob=preds_proba)
        # calculates the percentage of data used based on the instances and
        # features used from the original dataset
        train_percentage = pipeline_estimator.X_train.size / X_train.size
        samples_percentage = pipeline_estimator.X_train.shape[0] / X_train.shape[0]
        features_percentage = pipeline_estimator.X_train.shape[1] / X_train.shape[1]
        # calculate cpu time
        cpu_time = end_time - init_time
        # calculate balance metric
        balance_value = class_distribution_distance(
            classes_proportions=pipeline_estimator.y_train.value_counts(normalize=True).values,
            number_classes=y_train.nunique()
        )
        # calculate class proportions
        y_train_value_counts = pipeline_estimator.y_train.value_counts(normalize=True)
        proportion_per_class = []
        for class_name, proportion in y_train_value_counts.items():
            proportion_per_class.append((class_name, proportion))
        # create fitness
        # alpha * metric + beta * train_percentage + gama * time_spent
        if pipeline_config.get('time_norm', None) is None:
            fit = sum_components(
                pipeline_config=pipeline_config,
                pred_metric=pred_metric,
                train_percentage=train_percentage,
                cpu_time=cpu_time,
                balance_metric=balance_value)
        else:
            fit = sum_components_with_normalized_time(
                pipeline_config=pipeline_config,
                pred_metric=pred_metric,
                train_percentage=train_percentage,
                cpu_time=cpu_time,
                balance_metric=balance_value)
        
        fitness_params =  {
            'fitness' : fit,
            'search_metric' : pred_metric,
            'train_percentage' : train_percentage,
            'time_cpu' : cpu_time,
            'samples_percentage' : samples_percentage,
            'features_percentage' : features_percentage,
            'balance_metric' : balance_value,
            'proportion_per_class' : str(proportion_per_class),
            'individual_id' : individual_id,
            'data_processing_time' : pipeline_estimator.data_processing_time,
            'model_training_time' : pipeline_estimator.model_training_time,
            'prediction_time' : pipeline_estimator.prediction_time
        }

        if pipeline_config.get('flaml_ms', False):
            fitness_params['flaml_estimator'] = pipeline_estimator.model.best_estimator
            fitness_params['flaml_estimator_config'] = json.dumps(pipeline_estimator.model.best_config)
        return fitness_params
    
    except Exception as e:
        fitness_params = {
            'fitness' : 1,
            'search_metric' : 1,
            'train_percentage' : 1,
            'time_cpu' : 1,
            'samples_percentage' : 1,
            'features_percentage' : 1,
            'balance_metric' : 1,
            'proportion_per_class' : None,
            'individual_id' : individual_id,
            'data_processing_time' : 1,
            'model_training_time' : 1,
            'prediction_time' : 1
        }
        if pipeline_config.get('flaml_ms', False):
            fitness_params['flaml_estimator'] = None
            fitness_params['flaml_estimator_config'] = None
        return fitness_params

def sum_components(pipeline_config, pred_metric, train_percentage, cpu_time, balance_metric):
    """
    Calculates the fitness of the individual based on the different components and the weights associated

    Parameters:
    ----------
    pipeline_config : dict
        Dict with the weights for the components of the fitness

    pred_metric : float
        Prediction metric output, between 0 and 1

    train_percentage : float
        Percentage of train data used, between 0 and 1

    cpu_time : float
        Time the individual required to train, between 0 and infinite

    Returns:
    -------
        float
            Fitness
    """
    fitness = pipeline_config['alpha'] * pred_metric \
                + pipeline_config['beta'] * train_percentage \
                + pipeline_config['gama'] * cpu_time \
                + pipeline_config['delta'] * balance_metric
    return fitness


def sum_components_with_normalized_time(
        pipeline_config,
        pred_metric,
        train_percentage,
        cpu_time,
        balance_metric):
    """
    Calculates the fitness of the individual based on the different components and the weights associated,
    by normalising the time component of the fitness

    Parameters:
    ----------
    pipeline_config : dict
        Dict with the weights for the components of the fitness

    pred_metric : float
        Prediction metric output, between 0 and 1

    train_percentage : float
        Percentage of train data used, between 0 and 1

    cpu_time : float
        Time the individual required to train, between 0 and infinite

    Returns:
    -------
        float
            Fitness
    """
    time_component = 1 - (pipeline_config['time_norm'] / (1 + cpu_time))
    fitness = pipeline_config['alpha'] * pred_metric \
                + pipeline_config['beta'] * train_percentage \
                + pipeline_config['gama'] * time_component \
                + pipeline_config['delta'] * balance_metric
    return fitness

