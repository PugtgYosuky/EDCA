import matplotlib.pyplot as plt
from utils import *
import random as rnd
import json
import sys
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
import logging
from edca.utils import class_distribution_distance

# setup config
from sklearn import set_config
set_config(transform_output='pandas')

def disable(name):
    single_table_logger = logging.getLogger(name)
    handlers = single_table_logger.handlers
    single_table_logger.handlers = []
    try:
        yield
    finally:
        for handler in handlers:
            single_table_logger.addHandler(handler)


def disable_sdv_logger():
    for name in ['sdv', 'sdv.single_table', 'sdv.single_table.base', 'sdv.single_table.copulas', 'sdv.single_table.ctgan', 'sdv.single_table.copulagan']:
        disable(name)
plt.rcParams.update({
    'font.size': 12,       # Set font size
    'axes.labelsize': 'large',  # Set label size for x and y axes
    'axes.titlesize': 'x-large',  # Set title size
    'xtick.labelsize': 'medium',  # Set label size for x-axis ticks
    'ytick.labelsize': 'medium',  # Set label size for y-axis ticks
    'axes.grid': True,      # Show grid
    'grid.linestyle': '--',  # Set grid line style
    'grid.alpha': 0.5,      # Set grid transparency
    'grid.color': 'gray'    # Set grid color
})


def main(config, dataset_name, seed):
    disable_sdv_logger()
    # load dataset
    dataset = os.path.join('..', 'data', 'datasets', dataset_name)
    df = pd.read_csv(dataset)
    y = df.pop('class')

    if config['openml_splits']:
        metadata = pd.read_csv(os.path.join('..', 'data', 'metadata', 'classification_datasets_metadata.csv'))
        task = metadata.loc[metadata.name == dataset_name.split('.')[0], 'task'].values[0]
        task_id = task.split('/')[-1]
        data_splits = get_openml_splits(task_id)
    else:
        data_splits = get_kfold_splits(
            df, y, k=config.get(
                'kfold', 5), seed=seed)

    # create exp folder
    logs_path = os.path.join(config['save_path'], 'logs')
    if os.path.exists(logs_path) == False:
        os.makedirs(logs_path)
    exp = os.path.join(logs_path, f'exp_{datetime.now()}')
    os.makedirs(exp)
    os.makedirs(os.path.join(exp, 'data'))
    # save config
    config['dataset'] = dataset
    with open(os.path.join(exp, 'config.json'), 'w') as file:
        json.dump(config, file, indent=3)
        file.close()

    # kfold to evaluate the frameworks
    results = {}
    results['dataset'] = dataset
    for fold, (train_indexes, test_indexes) in enumerate(data_splits):
        if config.get('run-fold',None) is None or config.get('run-fold', None) == fold + 1:
            print('FOLD', fold + 1)
            # split data
            X_train = df.iloc[train_indexes]
            y_train = y.iloc[train_indexes]
            X_test = df.iloc[test_indexes]
            y_test = y.iloc[test_indexes]
            results['train_data'] = results.get('train_data', []) + [X_train.shape]
            results['test_data'] = results.get('test_data', []) + [X_test.shape]
            results['dataset_cdd'] = results.get('dataset_cdd', []) + [class_distribution_distance(np.array(y.value_counts(normalize=True)), y.nunique())]
            results['train_cdd'] = results.get('original_train_cdd', []) + [class_distribution_distance(np.array(y_train.value_counts(normalize=True)), y.nunique())]

            # test evo framework
            train_models(
                results,
                X_train,
                y_train,
                X_test,
                y_test,
                config,
                exp,
                fold,
                seed=seed)
            


if __name__ == '__main__':
    # read config
    with open(sys.argv[1], 'r') as file:
        config = json.load(file)

    datasets_default = [
        'mfeat-factors.csv',
        'Australian.csv',
        'credit-g.csv',
        'cnae-9.csv',
        'adult.csv',
        'bank-marketing.csv',
        'Amazon_employee_access.csv',
    ]

    datasets = config.get('datasets', datasets_default)

    # setup seeds
    seeds = [42, 384, 518, 522, 396, 400, 23, 791, 666, 283, 28, 298, 557, 309, 822, 569, 825, 185, 574, 325, 844, 90, 219, 864, 872, 618, 747, 365, 237, 767]
    
    if config.get('seed', None):
        seeds_to_run = [config.get('seed')]
    elif config.get('run_all_seeds', False) == False:
        seeds_to_run = [seeds[config.get('seed_pos', 0)]]
    elif config.get('run_all_seeds', False):
        seeds_to_run = seeds[config.get('seed_pos', 0):]
    else:
        seeds_to_run = seeds

    try:
        for seed in seeds_to_run:
            rnd.seed(seed)
            np.random.seed(seed)
            config['run_all_seeds'] = False
            config['seed_pos'] = int(np.argwhere(np.array(seeds) == seed))
            config['seed'] = seed
            for dataset in datasets:
                print('Running ', dataset)
                # to continue even if there was a problem in a dataset
                main(config, dataset, seed)
    except KeyboardInterrupt as e:
        print(e)
