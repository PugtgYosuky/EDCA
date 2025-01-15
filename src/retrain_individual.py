import pandas as pd
import numpy as np
import os
import json
import sys
from edca.estimator import PipelineEstimator
from utils import get_kfold_splits, get_openml_splits, append_metrics
import random as rnd
import pprint


def get_best_individual(path):
    evo_folds = os.listdir(os.path.join(path, 'evo'))
    evo_folds = [fold for fold in evo_folds if fold.startswith('evo')]
    bests_dfs = []
    for fold in evo_folds:
        aux = pd.read_csv(os.path.join(path, 'evo', fold, 'bests.csv'))
        aux['fold'] = fold
        bests_dfs.append(aux)
    df = pd.concat(bests_dfs, ignore_index=True)
    best_string = df.loc[df.fitness == df.fitness.min(), 'config'].values[0]
    return json.loads(best_string)

def run_experiment(dataset, seed, individual_config, pipeline_config, exp_path):
    # save individual

    task = metadata.loc[metadata.name == dataset.split('.')[0], 'task'].values[0]
    task_id = task.split('/')[-1]
    # get dataset
    df = pd.read_csv(os.path.join('..', 'datasets', dataset))
    y = df.pop('class')

    if config['openml_splits']:
        data_splits = get_openml_splits(task_id)
    else:
        data_splits = get_kfold_splits(df, y, k=config['kfold'], seed=seed)

    results = {}
    results['dataset'] = dataset

    for fold, (train_indexes, test_indexes) in enumerate(data_splits):
        print('FOLD', fold+1)

        X_train = df.iloc[train_indexes]
        y_train = y.iloc[train_indexes]
        X_test = df.iloc[test_indexes]
        y_test = y.iloc[test_indexes]

        # create pipeline estimator
        pipeline_estimator = PipelineEstimator(
            individual_config=individual_config, 
            pipeline_config=pipeline_config,
            use_sampling=False
        )
        
        # fit
        pipeline_estimator.fit(X_train, y_train)
        # predict
        y_pred = pipeline_estimator.predict(X_test)
        y_proba = pipeline_estimator.predict_proba(X_test)
        results = append_metrics(
            automl='evo',
            results= results,
            y_test=y_test,
            preds=y_pred,
            proba_preds=y_proba,
            sample_size=len(pipeline_estimator.X_train),
            train_data_size=len(X_train)
        )

    return results

if __name__ == "__main__":

    # read config
    with open(sys.argv[1], 'r') as file:
        config = json.load(file)
    
    # set seed
    seeds = config.get('seeds', 42)

    pipeline_config = None
    # get individual config
    if config['retrain_from_experiment']:
        print('retrain from experiment')
        # get best individual in the experiment
        individual_config = get_best_individual(config['retrain_from_experiment'])
        # get pipeline config
        with open(os.path.join(config['retrain_from_experiment'], 'evo', 'evo_fold1', 'pipeline_config.json')) as file:
            pipeline_config = json.load(file)
    else:
        with open(config['individual_config']) as file:
            individual_config = json.load(file)

    metadata = pd.read_csv(os.path.join( '..', 'metadata_tasks.csv'))
    # get data
    dataset = config['dataset_name']

    # set directory
    SAVE_PATH = os.path.join('retrain-results')
    if os.path.exists(SAVE_PATH) == False:
        os.makedirs(SAVE_PATH)
    DATASET_PATH = os.path.join(SAVE_PATH, dataset.split('.')[0])
    if os.path.exists(DATASET_PATH) == False:
        os.makedirs(DATASET_PATH)

    for seed in seeds:
        rnd.seed(seed)
        np.random.seed(seed)
        EXP_PATH = os.path.join(DATASET_PATH, f'exp_{len(os.listdir(DATASET_PATH))+1}')
        os.makedirs(EXP_PATH)
        results = run_experiment(dataset, seed, individual_config, pipeline_config, EXP_PATH)

        with open(os.path.join(EXP_PATH, 'config.json'), 'w') as file:
            json.dump(config, file, indent=2)

        with open(os.path.join(EXP_PATH, 'individual_config.json'), 'w') as file:
            json.dump(individual_config, file, indent=2)

        with open(os.path.join(EXP_PATH, 'results.json'), 'w') as file:
            json.dump(results, file, indent=2)

    


    