from sklearn import metrics
import numpy as np
import pandas as pd
import json
import time
import os
from datetime import datetime
from sklearn.base import clone  # to clone models
from sklearn.preprocessing import LabelEncoder  # to encode y
from sklearn.model_selection import StratifiedKFold, train_test_split, PredefinedSplit  # to divide data
import traceback
from edca.utils import class_distribution_distance

from edca.encoder import NpEncoder  # to encode dict to json
from edca.estimator import PipelineEstimator  # to instantiate pipeline
# openML framework
import openml

# autoML frameworks
from flaml import AutoML
from tpot import TPOTClassifier
from tpot.export_utils import set_param_recursive
from edca.evodata import DataCentricAutoML

# setup sklearn to use pandas.DataFrame as output
from sklearn import set_config
set_config(transform_output='pandas')


def add_value_to_dict(dictionary, key, value):
    dictionary[key] = dictionary.get(key, []) + [value]


def append_metrics(
        automl_framework,
        results,
        y_test,
        preds,
        proba_preds,
        final_data_size,
        original_data_size,
        cdd=None, 
        error=False):
    # write as none if it has an error
    variables_names = [
        '_sample_%',
        '_samples_used',
        '_features_%',
        '_features_used',
        '_data_%',
        'data_size',
        '_cdd',
        '_roc_auc',
        '_logloss',
        '_f1',
        '_mcc',
        '_roc_auc_per_sample',
        '_logloss_per_sample',
        '_f1_per_sample',
        '_mcc_per_sample',
        '_tp',
        '_tn',
        '_fp',
        '_fn']
    if error:
        for var in variables_names:
            add_value_to_dict(results, f'{automl_framework}{var}', None)
        return
    # calculate data percentages
    final_sample_size, final_features_size = final_data_size
    original_sample_size, original_features_size = original_data_size
    data_percentage = (final_sample_size * final_features_size) / \
        (original_sample_size * original_features_size)
    sample_percentage = final_sample_size / original_sample_size
    features_percentage = final_features_size / original_features_size

    add_value_to_dict(
        results,
        f'{automl_framework}_sample_%',
        sample_percentage)
    add_value_to_dict(
        results,
        f'{automl_framework}_features_%',
        features_percentage)
    add_value_to_dict(results, f'{automl_framework}_data_%', data_percentage)
    add_value_to_dict(results, f'{automl_framework}_cdd', cdd)
    add_value_to_dict(
        results, f'{automl_framework}_final_data_size', [
            final_sample_size, final_features_size])
    add_value_to_dict(results, f'{automl_framework}_original_data_size', [
                      original_sample_size, original_features_size])

    # calculate metrics

    f1 = metrics.f1_score(y_test, preds, average='weighted')
    add_value_to_dict(results, f'{automl_framework}_f1', f1)
    add_value_to_dict(
        results,
        f'{automl_framework}_f1_per_data',
        f1 / data_percentage)
    add_value_to_dict(
        results,
        f'{automl_framework}_f1_per_sample',
        f1 / sample_percentage)
    add_value_to_dict(
        results,
        f'{automl_framework}_f1_per_feature',
        f1 / features_percentage)
    mcc = metrics.matthews_corrcoef(y_test, preds)
    add_value_to_dict(results, f'{automl_framework}_mcc', mcc)
    add_value_to_dict(
        results,
        f'{automl_framework}_mcc_per_data',
        mcc / data_percentage)
    add_value_to_dict(
        results,
        f'{automl_framework}_mcc_per_sample',
        mcc / sample_percentage)
    add_value_to_dict(
        results,
        f'{automl_framework}_mcc_per_feature',
        mcc / features_percentage)

    # calculate # class dependent metrics
    if len(y_test.unique()) == 2:
        # binary problem
        roc_auc = metrics.roc_auc_score(y_test, proba_preds[:, 1])
        log_loss = metrics.log_loss(y_test, proba_preds)

        tn, fp, fn, tp = metrics.confusion_matrix(y_test, preds).ravel()
        add_value_to_dict(results, f'{automl_framework}_tp', tp)
        add_value_to_dict(results, f'{automl_framework}_fp', fp)
        add_value_to_dict(results, f'{automl_framework}_tn', tn)
        add_value_to_dict(results, f'{automl_framework}_fn', fn)
    else:
        # multiclass problem
        roc_auc = metrics.roc_auc_score(
            y_test, proba_preds, average='macro', multi_class="ovr")
        log_loss = metrics.log_loss(y_test, proba_preds)

    add_value_to_dict(results, f'{automl_framework}_roc_auc', roc_auc)
    add_value_to_dict(
        results,
        f'{automl_framework}_roc_auc_per_data',
        roc_auc / data_percentage)
    add_value_to_dict(
        results,
        f'{automl_framework}_roc_auc_per_sample',
        roc_auc / sample_percentage)
    add_value_to_dict(
        results,
        f'{automl_framework}_roc_auc_per_feature',
        roc_auc /
        features_percentage)
    add_value_to_dict(results, f'{automl_framework}_logloss', log_loss)
    add_value_to_dict(
        results,
        f'{automl_framework}_logloss_per_data',
        log_loss /
        data_percentage)
    add_value_to_dict(
        results,
        f'{automl_framework}_logloss_per_sample',
        log_loss /
        sample_percentage)
    add_value_to_dict(
        results,
        f'{automl_framework}_logloss_per_feature',
        log_loss /
        features_percentage)


def get_openml_splits(task_id):
    task = openml.tasks.get_task(task_id)
    split = task.download_split()
    split_data = split.split

    data_splits = []
    for fold in range(len(split_data[0])):
        data_splits.append(
            (
                split_data[0][fold][0].train,
                split_data[0][fold][0].test
            )
        )

    return data_splits


def get_kfold_splits(df, y, k=5, seed=42):
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    data_splits = []
    for fold_data in kfold.split(df, y):
        data_splits.append(fold_data)
    return data_splits


def retrain_edca_from_data(
        name,
        automl,
        X_train,
        y_train,
        X_test,
        y_test,
        results):
    individual = automl.get_best_individual().copy()
    if 'sample' in individual:
        _ = individual.pop('sample')  # remove sample component
    if 'features' in individual:
        _ = individual.pop('features')  # remove features component

    pipeline_estimator = PipelineEstimator(
        individual_config=individual,
        pipeline_config=automl.pipeline_config,
    )

    pipeline_estimator.fit(X_train, y_train)
    preds = pipeline_estimator.predict(X_test)
    proba_preds = pipeline_estimator.predict_proba(X_test)

    append_metrics(
        automl_framework=name,
        results=results,
        y_test=y_test,
        preds=preds,
        proba_preds=proba_preds,
        final_data_size=X_train.shape,
        original_data_size=X_train.shape,
        cdd=class_distribution_distance(np.array(y_train.value_counts(normalize=True)), y_train.nunique())
    )


def evo_train(
        results,
        X_train,
        y_train,
        X_test,
        y_test,
        config,
        evo_path,
        fold):

    has_error = False
    start = time.time()  # start counter
    evo_automl = DataCentricAutoML(
        task=config.get('task', 'classification'),
        seed=config.get('seed', 42),
        metric=config.get('metric', 'mcc'),
        population_size=config.get('population'),
        prob_mutation=config.get('prob_mutations', 0.3),
        prob_mutation_model=config.get('prob_mutation_model', 0.5),
        prob_crossover=config.get('prob_crossover', 0.7),
        tournament_size=config.get('tournament_size', 3),
        elitism_size=config.get('elitism_size', 1),
        n_iterations=config.get('n_iterations', 50),
        binary_sampling_component=config.get('binary_sampling_component', True),
        automatic_data_optimization=config.get('automatic_data_optimization', False),
        use_sampling=config.get('sampling', True),
        use_feature_selection=config.get('feature_selection', False),
        use_data_augmentation=config.get('data_augmentation', False),
        sampling_start=config.get('sampling_start', 0),
        alpha=config.get('alpha'),
        beta=config.get('beta'),
        gama=config.get('gama'),
        delta=config.get('delta'),
        verbose=config.get('verbose', -1),
        time_norm=config.get('time_norm', None),
        time_budget=config.get('time_budget', -1),
        class_balance_mutation=config.get('class_balance_mutation', False),
        uniform_crossover=config.get('uniform_crossover', True),
        mutation_factor=config.get('mutation_factor', 0.5),
        log_folder_name=os.path.join(evo_path, f'evo_fold{fold+1}'),
        n_jobs=config.get('n_jobs', 1),
        patience=config.get('patience', None),
        early_stop=config.get('early_stop', None),
        validation_size=config.get('validation_size', 0.25),
        mutation_size_neighborhood=config.get('mutation_size_neighborhood', 10),
        mutation_percentage_change=config.get('mutation_percentage_change', 0.1),
    )

    try:
        evo_automl.fit(X_train, y_train)
        end = time.time()  # end counter
        results['evo_time'] = results.get('evo_time', []) + [end - start]

        evo_preds = evo_automl.predict(X_test)  # prediction
        evo_proba_preds = evo_automl.predict_proba(
            X_test)  # prediction probabilities

        _, final_y = evo_automl.get_final_data()
        final_data_shape = evo_automl.get_selected_data_shape()

        append_metrics(
            automl_framework='evo',
            results=results,
            y_test=y_test,
            preds=evo_preds,
            proba_preds=evo_proba_preds,
            final_data_size=final_data_shape,
            original_data_size=X_train.shape,
            cdd=class_distribution_distance(np.array(final_y.value_counts(normalize=True)), y_train.nunique())
        )

        # save predictions
        predictions_df = pd.DataFrame(
            evo_proba_preds, columns=[
                f'y_proba_{i}' for i in range(
                    evo_proba_preds.shape[1])])
        predictions_df.insert(0, 'y_test', list(y_test))
        predictions_df.insert(1, 'y_pred', evo_preds)
        predictions_df.to_csv(
            os.path.join(
                evo_path,
                'predictions',
                f'evo_predictions_{fold+1}.csv'),
            index=False)

        add_value_to_dict(
            results,
            'evo_num_iterations',
            evo_automl.search_algo.get_number_iterations())
        add_value_to_dict(
            results, 
            'evo_num_pipelines_tested',
            evo_automl.search_algo.get_number_pipelines_tested())
        add_value_to_dict(
            results,
            'evo_best',
            evo_automl.get_best_individual())

        # retrain with the hole dataset
        if config.get('sampling') or config.get('feature_selection'):
            retrain_edca_from_data(
                name='evo_all_data',
                automl=evo_automl,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                results=results
            )
    except KeyboardInterrupt:
        raise KeyboardInterrupt('Ctrl-C pressed')

    except Exception as error:
        print('Error in EDCA: ', error)
        traceback.print_exc()
        has_error = True

        add_value_to_dict(results, 'evo_time', None)
        add_value_to_dict(results, 'evo_num_iterations', None)
        add_value_to_dict(results, 'evo_best', None)

        append_metrics(
            automl_framework='evo',
            results=results,
            y_test=y_test,
            preds=None,
            proba_preds=None,
            final_data_size=None,
            original_data_size=None,
            cdd=None,
            error=True)
    return evo_automl, has_error


def get_flaml_info(filename):
    # loads the sample size of the best estimator from log file
    with open(filename, 'r') as file:
        lines = file.readlines()
        best_estimator = json.loads(lines[-1])
        best_id = best_estimator['curr_best_record_id']
        sample_size = json.loads(lines[best_id])['sample_size']
        num_iterations = len(lines) - 1
    return sample_size, num_iterations


def retrain_flaml_from_data(automl, X_train, y_train, X_test):

    # clone original trained model
    model = clone(automl.model)
    # transform data
    train_X = automl._state.task.preprocess(X_train, automl._transformer)
    test_X = automl._state.task.preprocess(X_test, automl._transformer)

    model.fit(train_X, y_train)

    preds = model.predict(test_X)
    proba_preds = model.predict_proba(test_X)
    return preds, proba_preds


def flaml_train(
        results,
        X_train,
        y_train,
        X_test,
        y_test,
        config,
        flaml_path,
        fold):

    # encode y_train
    y_encoder = LabelEncoder()
    train_y = y_encoder.fit_transform(y_train)
    x_train, x_val, y_train, y_val = train_test_split(
        X_train, train_y, 
        stratify=train_y, 
        shuffle=True, 
        test_size=config.get('validation_size', 0.1),
        random_state=config.get('seed', 42)
    )


    # setup error
    has_error = False
    final_data_rows = None

    start = time.time()  # start counter
    log_file = os.path.join(flaml_path, f'flaml_fold{fold+1}.log')

    settings = dict(
        metric=flaml_metric(config.get('metric', 'mcc')),
        task='classification',
        eval_method='holdout',
        time_budget=config.get('time_budget', -1),
        hpo_method=config.get('hpo_method', 'cfo'),
        log_file_name=log_file,
        log_type='all',
        early_stop=config.get('early_stop', None)!=None,
        sample=config.get('sampling', True),
        retrain_full=not config.get('sampling', True),
        split_ration=config.get('validation_size', 0.9),
        n_jobs=config.get('n_jobs', 1),
        estimator_list=list(sorted(['lgbm', 'xgboost', 'xgb_limitdepth', 'rf', 'lrl1', 'lrl2', 'kneighbor', 'extra_tree'])),
        seed = config.get('seed', 42),
    )

    if config.get('time_budget', -1) == -1:
        # use iterations instead od time budget
        settings['max_iter'] = config.get('n_iterations', 50) # remove: * config.get('population', 25)

    flaml_automl = AutoML(**settings)
    try:
        flaml_automl.fit(
            X_train=x_train, 
            y_train=y_train,
            X_val=x_val,
            y_val=y_val)
        end = time.time()  # end counter

        add_value_to_dict(results, 'flaml_time', end - start)
        flaml_preds = y_encoder.inverse_transform(flaml_automl.predict(X_test))
        flaml_proba_preds = flaml_automl.predict_proba(X_test)
        final_data_rows, num_iterations = get_flaml_info(log_file)

        # save results
        append_metrics(
            automl_framework='flaml',
            results=results,
            y_test=y_test,
            preds=flaml_preds,
            proba_preds=flaml_proba_preds,
            final_data_size=[final_data_rows, X_train.shape[1]],
            original_data_size=X_train.shape,
            cdd=class_distribution_distance(np.array(y_train.value_counts(normalize=True)), y_train.nunique())
        )
        # save predictions
        predictions_df = pd.DataFrame(
            flaml_proba_preds, columns=[
                f'y_proba_{i}' for i in range(
                    flaml_proba_preds.shape[1])])
        predictions_df.insert(0, 'y_test', y_test)
        predictions_df.insert(1, 'y_pred', flaml_preds)
        predictions_df.to_csv(
            os.path.join(
                flaml_path,
                'predictions',
                f'flaml_predictions_{fold+1}.csv'),
            index=False)

        # add metrics
        add_value_to_dict(results, 'flaml_num_iterations', num_iterations)
        add_value_to_dict(
            results,
            'flaml_best_learner',
            flaml_automl.best_estimator)
        add_value_to_dict(
            results,
            'flaml_best_config',
            flaml_automl.best_config)

        if config.get('sampling', True):
            all_data_preds, all_data_proba_preds = retrain_flaml_from_data(
                automl=flaml_automl,
                X_train=X_train,
                y_train=train_y,
                X_test=X_test
            )
            all_data_preds = y_encoder.inverse_transform(all_data_preds)

            append_metrics(
                automl_framework='flaml_all_data',
                results=results,
                y_test=y_test,
                preds=all_data_preds,
                proba_preds=all_data_proba_preds,
                final_data_size=X_train.shape,
                original_data_size=X_train.shape,
                cdd=class_distribution_distance(np.array(train_y.value_counts(normalize=True)), train_y.nunique())
            )

    except KeyboardInterrupt:
        raise KeyboardInterrupt('Ctrl-C pressed')

    except Exception as error:
        print('Error in FLAML: ', error)
        has_error = True

        add_value_to_dict(results, 'flaml_time', None)
        add_value_to_dict(results, 'flaml_num_iterations', None)
        add_value_to_dict(results, 'flaml_best_learner', None)
        add_value_to_dict(results, 'flaml_best_config', None)

        append_metrics(
            automl_framework='flaml',
            results=results,
            y_test=y_test,
            preds=None,
            proba_preds=None,
            final_data_size=None,
            original_data_size=None,
            cdd=None,
            error=True)

        if config.get('sampling', True):
            append_metrics(
                automl_framework='flaml_all_data',
                results=results,
                y_test=y_test,
                preds=None,
                proba_preds=None,
                final_data_size=None,
                original_data_size=None,
                cdd=class_distribution_distance(np.array(train_y.value_counts(normalize=True)), train_y.nunique()),
                error=True)
    return flaml_automl, has_error, final_data_rows, y_encoder


def tpot_train(
        results,
        X_train,
        y_train,
        X_test,
        y_test,
        config,
        tpot_path,
        fold,
        seed):
    y_encoder = LabelEncoder()
    train_y = y_encoder.fit_transform(y_train)

    # calculate indexes for train test
    aux_train, _, _, _ = train_test_split(
        X_train, y_train, 
        stratify=y_train, 
        shuffle=True, 
        test_size=config.get('validation_size', 0.1),
        random_state=config.get('seed', 42)
    )

    indexes = pd.Series([0] * len(X_train))
    indexes.loc[X_train.index.isin(list(aux_train.index))] = -1

    ps = PredefinedSplit(test_fold=list(indexes))

    start = time.time()  # start counter

    log_file = os.path.join(tpot_path, f'tpot_fold{fold+1}.log')

    has_error = False

    feature_selection_config = {
        'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': np.arange(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }
    }

    # tpot search space
    classifiers_config = {
        'sklearn.ensemble.ExtraTreesClassifier': {
            'n_estimators': [100],
            'criterion': ["gini", "entropy"],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': np.arange(2, 21),
            'min_samples_leaf': np.arange(1, 21),
            'bootstrap': [True, False]
        },
        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': [100],
            'criterion': ["gini", "entropy"],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': np.arange(2, 21),
            'min_samples_leaf': np.arange(1, 21),
            'bootstrap': [True, False]
        },
        'sklearn.neighbors.KNeighborsClassifier': {
            'n_neighbors': np.arange(1, 101),
            'weights': ["uniform", "distance"],
            'p': [1, 2]
        },
        'sklearn.linear_model.LogisticRegression': {
            'penalty': ["l1", "l2"],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'dual': [True, False]
        },
        'xgboost.XGBClassifier': {
            'n_estimators': [100],
            'max_depth': np.arange(1, 11),
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'subsample': np.arange(0.05, 1.01, 0.05),
            'min_child_weight': np.arange(1, 21),
            'n_jobs': [1],
            'verbosity': [0]
        },
        'sklearn.preprocessing.MinMaxScaler': {
        },
        'sklearn.preprocessing.RobustScaler': {
        },
        'sklearn.preprocessing.StandardScaler': {
        },
        'tpot.builtins.OneHotEncoder': {
            'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
            'sparse': [False],
            'threshold': [10]
        },
        'lightgbm.LGBMClassifier': {
            'n_estimators': np.arange(4, 1000),
            'num_leaves': np.arange(4, 1000),
            'min_child_samples': np.arange(2, 129),
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'colsample_bytree': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'verbosity': [-1]
        }
    }

    if config.get('feature_selection', False):
        # add feature selection to config
        classifiers_config.update(feature_selection_config)

    settings = {
        'population_size': config.get(
            'population',
            25),
        'log_file': log_file,
        'scoring': metrics.make_scorer(
            tpot_metric(
                config.get(
                    'metric',
                    'mcc')),
            greater_is_better=False),
        'random_state': seed,
        'verbosity': 3,
        'config_dict': classifiers_config,
        'template' : config.get('tpot_template', None),
        'early_stop': config.get('early_stop', None),
        'n_jobs': config.get(
            'n_jobs',
            1),
        'cv': ps
    }

    time_budget = config.get('time_budget', -1)
    if time_budget == -1:
        settings['generations'] = config.get('n_iterations', 50)
    else:
        settings['max_time_mins'] = int(time_budget / 60)

    tpot_automl = TPOTClassifier(**settings)
    try:
        tpot_automl.fit(X_train, train_y)
        end = time.time()  # end counter
        add_value_to_dict(results, 'tpot_time', end - start)
        tpot_preds = y_encoder.inverse_transform(tpot_automl.predict(X_test))
        tpot_proba_preds = tpot_automl.predict_proba(X_test)
        # save data used
        if config.get('tpot_template', None) is not None and config.get('feature_selection', False):
            fitted_pipeline = tpot_automl.fitted_pipeline_
            transform_pipeline = fitted_pipeline[:-1]
            X_train_transformed = transform_pipeline.transform(X_train)
            X_test_transformed = transform_pipeline.transform(X_test)
            X_train_transformed.to_csv(os.path.join(tpot_path, f'transformed_train_data_fold_{fold+1}.csv'))
            X_test_transformed.to_csv(os.path.join(tpot_path, f'transformed_test_data_fold_{fold+1}.csv'))
        else:
            X_train_transformed = X_train
        # save metrics achieved
        append_metrics(
            automl_framework='tpot',
            results=results,
            y_test=y_test,
            preds=tpot_preds,
            proba_preds=tpot_proba_preds,
            final_data_size=X_train_transformed.shape,
            original_data_size=X_train.shape,
            cdd=class_distribution_distance(np.array(y_train.value_counts(normalize=True)), y_train.nunique())
        )

        # save evaluated individuals
        evaluated_individuals = tpot_automl.evaluated_individuals_
        with open(os.path.join(tpot_path, f'evaluated_individuals_fold_{fold+1}.json'), 'w') as file:
            json.dump(evaluated_individuals, file, cls=NpEncoder, indent=3)
        # save predictions
        predictions_df = pd.DataFrame(
            tpot_proba_preds, columns=[
                f'y_proba_{i}' for i in range(
                    tpot_proba_preds.shape[1])])
        predictions_df.insert(0, 'y_test', y_test)
        predictions_df.insert(1, 'y_pred', tpot_preds)
        predictions_df.to_csv(
            os.path.join(
                tpot_path,
                'predictions',
                f'tpot_predictions_{fold+1}.csv'),
            index=False)

        # export TPOT best pipeline to json
        # pipeline_json = []
        # for name, operator in tpot_automl.fitted_pipeline_.steps:
        #     pipeline_json.append({
        #         operator.__class__.__name__: operator.get_params()
        #     })
        #     print(name, type(operator.get_params))

        add_value_to_dict(results, 'tpot_best_pipeline', str(tpot_automl.fitted_pipeline_.steps))
        total_evaluated = tpot_automl._pbar.n
        num_generations = int(total_evaluated / config.get('population', 25))-1
        add_value_to_dict(results, 'tpot_num_iterations', num_generations)
        add_value_to_dict(results, 'tpot_total_evaluated', total_evaluated)

        print('>> TPOT: export')
        # save best pipeline
        tpot_automl.export(
            os.path.join(
                tpot_path,
                f'best_pipeline_fold{fold+1}.py'))

    except KeyboardInterrupt:
        raise KeyboardInterrupt('Ctrl-C pressed')
    except Exception as error:
        print('Error in TPOT: ', error)
        has_error = True

        add_value_to_dict(results, 'tpot_time', None)
        add_value_to_dict(results, 'tpot_best_pipeline', None)
        add_value_to_dict(results, 'tpot_num_iterations', None)

        append_metrics(
            automl_framework='tpot',
            results=results,
            y_test=y_test,
            preds=None,
            proba_preds=None,
            final_data_size=None,
            original_data_size=None,
            cdd=None,
            error=True)

    return tpot_automl, has_error, y_encoder


def train_models(
        results,
        X_train,
        y_train,
        X_test,
        y_test,
        config,
        path,
        fold,
        seed):

    # save datasets
    if config.get('save_data', False):
        train_data = X_train.copy()
        train_data['class'] = y_train
        train_data.to_csv(
            os.path.join(
                path,
                'data',
                f'train_data_fold{fold+1}.csv'),
            index=False)
        test_data = X_test.copy()
        test_data['class'] = y_test
        test_data.to_csv(
            os.path.join(
                path,
                'data',
                f'test_data_fold{fold+1}.csv'),
            index=False)

    evo_error = False
    tpot_error = False
    flaml_error = False
    # train EDCA
    if config.get('train_evo', True):
        evo_path = os.path.join(path, 'evo')
        if not os.path.exists(evo_path):
            os.makedirs(evo_path)
            os.makedirs(os.path.join(evo_path, 'predictions'))
        evo_automl, evo_error = evo_train(
            results=results,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=config,
            evo_path=evo_path,
            fold=fold
        )
        save_results(results, path)

    # train flaml
    if config.get('train_flaml', True):
        flaml_path = os.path.join(path, 'flaml')
        if not os.path.exists(flaml_path):
            os.makedirs(flaml_path)
            os.makedirs(os.path.join(flaml_path, 'predictions'))
        flaml_automl, flaml_error, flaml_sample_size, flaml_y_encoder = flaml_train(
            results=results,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=config,
            flaml_path=flaml_path,
            fold=fold
        )
        save_results(results, path)

    if config.get('train_tpot', True):
        tpot_path = os.path.join(path, 'tpot')
        if not os.path.exists(tpot_path):
            os.makedirs(tpot_path)
            os.makedirs(os.path.join(tpot_path, 'predictions'))

        tpot_automl, tpot_error, tpot_y_encoder = tpot_train(
            results=results,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=config,
            tpot_path=tpot_path,
            fold=fold,
            seed=seed
        )
        save_results(results, path)

    if config.get(
        'train_evo', True) and config.get(
        'train_flaml', True) and (
            not evo_error) and (
                not flaml_error):
        # test combinations

        # flaml with EDCA data
        x_data, y_data = evo_automl.get_best_sample_data()

        # encode y
        y_data = flaml_y_encoder.transform(y_data)

        print('FLAML with EDCA data')
        flaml_preds, flaml_proba_preds = retrain_flaml_from_data(
            automl=flaml_automl,
            X_train=x_data,
            y_train=y_data,
            X_test=X_test
        )
        flaml_preds = flaml_y_encoder.inverse_transform(flaml_preds)
        append_metrics(
            automl_framework='flaml_with_evo_samples',
            results=results,
            y_test=y_test,
            preds=flaml_preds,
            proba_preds=flaml_proba_preds,
            final_data_size=x_data.shape,
            original_data_size=X_train.shape,
            cdd=class_distribution_distance(np.array(y_data.value_counts(normalize=True)), y_train.nunique())
        )

        # train EDCA with FLAML data
        # get data
        x_data, _, y_data, _ = train_test_split(
            X_train, y_train, stratify=y_train, shuffle=True, train_size=flaml_sample_size)
        print('EDCA with FLAML data')
        retrain_edca_from_data(
            name='evo_with_flaml_samples',
            automl=evo_automl,
            X_train=x_data,
            y_train=y_data,
            X_test=X_test,
            y_test=y_test,
            results=results
        )
        save_results(results, path)

    if config.get(
        'train_evo', True) and config.get(
        'train_tpot', True) and (
            not evo_error) and (
                not tpot_error):
        # TPOT with EDCA data
        x_data, y_data = evo_automl.get_best_sample_data()
        print('TPOT with EDCA data')
        tpot_preds, tpot_proba_preds = retrain_tpot_from_data(
            automl=tpot_automl,
            X_train=x_data,
            y_train=y_data,
            X_test=X_test,
            tpot_y_encoder=tpot_y_encoder
        )
        append_metrics(
            automl_framework='tpot_with_evo_samples',
            results=results,
            y_test=y_test,
            preds=tpot_preds,
            proba_preds=tpot_proba_preds,
            final_data_size=x_data.shape,
            original_data_size=X_train.shape,
            cdd=class_distribution_distance(np.array(y_data.value_counts(normalize=True)), y_train.nunique())
        )
        save_results(results, path)
    save_results(results, path)


def retrain_tpot_from_data(automl, X_train, y_train, X_test, tpot_y_encoder):
    # clone original trained model
    print('>>>TPOT retrain: clone')
    model = clone(automl.fitted_pipeline_)
    train_y = tpot_y_encoder.transform(y_train)
    print('<<< TPOT retrain: clone')
    model.fit(X_train, train_y)

    preds = tpot_y_encoder.inverse_transform(model.predict(X_test))
    proba_preds = model.predict_proba(X_test)
    return preds, proba_preds


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
        return flaml_f1_metric
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


def save_results(results, exp):
    with open(os.path.join(exp, 'results.json'), 'w') as file:
        json.dump(results, file, indent=3, cls=NpEncoder)
