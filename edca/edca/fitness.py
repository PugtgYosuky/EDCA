import time
import json
from edca.estimator import PipelineEstimator
from edca.utils import class_distribution_distance, fairness_metric
import pandas as pd
from sklearn import metrics
from edca.utils import debug_print, DEBUG_MODE

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
        if 'individual_id' in individual:
            del individual['individual_id']
        pipeline_estimator = PipelineEstimator(
            individual_config=individual,
            pipeline_config=pipeline_config,
            individual_id=individual_id,
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
        normalized_time = 1 - (1 / (1 + cpu_time))
        # calculate balance metric
        proportions = pipeline_estimator.y_train.value_counts(normalize=True)
        balance_value = class_distribution_distance(
            classes_proportions=proportions.values,
            number_classes=y_train.nunique()
        )

        # calculate fairness metrics
        if pipeline_config.get('fairness_params', None):
            fairness, fair_metrics = fairness_metric(
                y_true = y_val,
                y_pred = preds,
                y_proba = preds_proba[:, 1],
                X_test = X_val,
                fairness_params = pipeline_config['fairness_params']
            )
            
        else:
            fairness, fair_metrics = None, {}
        # calculate fitness
        fitness_value = pipeline_config['fitness_params']['metric'] * pred_metric \
            + pipeline_config['fitness_params']['data_size'] * train_percentage \
            + pipeline_config['fitness_params']['training_time'] * normalized_time \
            + pipeline_config['fitness_params']['balance_metric'] * balance_value \
            + pipeline_config['fitness_params']['fairness_metric'] * (fairness if fairness is not None else 1)
        
        fitness_params =  {
            'fitness' : round(fitness_value, 3),
            'search_metric' : pred_metric,
            'train_percentage' : train_percentage,
            'time_cpu' : cpu_time,
            'samples_percentage' : samples_percentage,
            'features_percentage' : features_percentage,
            'balance_metric' : balance_value,
            'proportion_per_class' : str(proportions.to_dict()),
            'fairness_metric' : fairness,

            'individual_id' : individual_id,
            'data_processing_time' : pipeline_estimator.data_processing_time,
            'model_training_time' : pipeline_estimator.model_training_time,
            'prediction_time' : pipeline_estimator.prediction_time
        }
        if y_val.nunique()==2:
            tn, fp, fn, tp = metrics.confusion_matrix(y_val, preds).ravel()
            cm = {
            'tn' : tn,
            'fp' : fp,
            'fn' : fn,
            'tp' : tp,
            }
            fitness_params.update(cm)
        # update with fairness metrics
        fitness_params.update(fair_metrics)

        if pipeline_config.get('flaml_ms', False):
            fitness_params['flaml_estimator'] = pipeline_estimator.model.best_estimator
            fitness_params['flaml_estimator_config'] = json.dumps(pipeline_estimator.model.best_config)
        return fitness_params
    
    except Exception as e:
        debug_print(e)
        fitness_params = {
            'fitness' : 1,
            'search_metric' : 1,
            'train_percentage' : 1,
            'time_cpu' : 1,
            'samples_percentage' : 1,
            'features_percentage' : 1,
            'balance_metric' : 1,
            'proportion_per_class' : None,
            'fairness_metric' : 1,
            'demographic_parity_difference' : 1,
            'equal_opportunity_difference' : 1,
            'equalized_odds_difference' : 1,
            'individual_id' : individual_id,
            'data_processing_time' : 1,
            'model_training_time' : 1,
            'prediction_time' : 1
        }
        if pipeline_config.get('flaml_ms', False):
            fitness_params['flaml_estimator'] = None
            fitness_params['flaml_estimator_config'] = None
        return fitness_params
    
