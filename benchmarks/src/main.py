
"""
Source code for running the benchmarks using cross-validation
"""

from copy import deepcopy
import matplotlib.pyplot as plt
from utils import *
import random as rnd
import json
import sys
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
from edca.encoder import NpEncoder
import logging
from edca.utils import class_distribution_distance
from datetime import datetime

# setup config
from sklearn import set_config
set_config(transform_output='pandas')

DEFAULT_SAVE_DIR = 'logs'
DEFAULT_DATASETS_SRC_DIR = 'data/datasets'



# attempt to import served dependent variables
try:
    from config_variables import SAVE_DIR, DATASETS_SRC_DIR
except:
    SAVE_DIR = DEFAULT_SAVE_DIR
    DATASETS_SRC_DIR = DEFAULT_DATASETS_SRC_DIR

def disable(name):
    single_table_logger = logging.getLogger(name)
    handlers = single_table_logger.handlers
    single_table_logger.handlers = []
    try:
        yield
    finally:
        for handler in handlers:
            single_table_logger.addHandler(handler)


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
    # load dataset
    train_path = os.path.join(DATASETS_SRC_DIR, config.get('train_dataset', dataset_name))
    test_path = os.path.join(DATASETS_SRC_DIR, config['test_dataset'])
    
    label_col = config.get('label_col', 'Class')
    drop_cols = config.get('drop_cols', [])

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    y_train = df_train.pop(label_col)
    y_test = df_test.pop(label_col)

    # drop Processo
    for c in drop_cols:
        if c in df_train.columns:
            df_train = df_train.drop(columns=[c])
        if c in df_test.columns:
            df_test = df_test.drop(columns=[c])
    
    df_test = df_test[df_train.columns]

    
    # create experiments folder divided by dataset name
    experiment_path = os.path.join(SAVE_DIR, config['save_path'], f'exp_{datetime.now()}')
    os.makedirs(experiment_path, exist_ok=True)
    # os.makedirs(os.path.join(experiment_path, 'data'))
    # save config
    config['train_path'] = train_path
    config['test_path'] = test_path
    if config.get('fairness_params', None):
        config['fairness_params'] = config['fairness_params'][dataset_name]
    
    with open(os.path.join(experiment_path, 'config.json'), 'w') as file:
        # import pprint
        # pprint.pprint(config)
        json.dump(config, file, indent=3, cls=NpEncoder)
        file.close()

    # kfold to evaluate the frameworks
    results = {}
    results['train_path'] = train_path
    results['test_path'] = test_path

    results['run_info'] = {
        'train_data': df_train.shape,
        'test_data': df_test.shape,
        'train_cdd': class_distribution_distance(np.array(y_train.value_counts(normalize=True)), y_train.nunique()),
        'test_cdd': class_distribution_distance(np.array(y_test.value_counts(normalize=True)), y_test.nunique()),
    }
    

    
    # test evo framework
    train_models(
        results['run_info'],
                df_train,
                y_train,
                df_test,
                y_test,
                config,
                experiment_path,
                0,
                save_results=save_results(results, experiment_path),
                seed=seed
                )

    
            
            
def update_config_params(config):
    if 'alpha' in config:
        # old config setup -> update to new config
        fitness_params = {
            'metric' : config.pop('alpha', 1.0),
            'data_size' : config.pop('beta', 0.0),
            'training_time' : config.pop('gamma', 0.0),
            'balance_metric' : config.pop('delta', 0.0),
        }
        config['fitness_params'] = fitness_params
    return config

def run_config(config_path, datasets_default):
    print('>>>>>>>>> Running config:', config_path)
    # read config
    with open(config_path, 'r') as file:
        config = json.load(file)

    config = update_config_params(config)
    datasets = config.get('dataset', datasets_default)

    # setup seeds
    #seeds = [42, 384, 518, 522, 396, 400, 23, 791, 666, 283, 28, 298, 557, 309, 822, 569, 825, 185, 574, 325, 844, 90, 219, 864, 872, 618, 747, 365, 237, 767]
    seeds = [864, 872, 618, 747, 365, 237, 767]
    #seeds = [42, 384, 518, 522, 396, 400, 23, 791, 666, 283]
    
    if len(datasets) == 1 and 'deliveries' in datasets[0]:
        #seeds = [123, 987, 456, 789, 321, 654, 768, 234, 567, 890, 432, 765, 109, 876, 543, 210, 897, 345, 678, 901, 1234, 5678, 9012, 3456, 7890, 2345, 6789, 1263, 4567, 8901]
        seeds = [3456, 7890, 2345, 6789, 1263, 4567, 8901]

    if config.get('seed', None):
        seeds_to_run = [config['seed']]
    elif config.get('run_all_seeds', False) == False:
        seeds_to_run = [seeds[config.get('seed_pos', 0)]]
    elif config.get('run_all_seeds', False):
        seeds_to_run = seeds[config.get('seed_pos', 0):config.get('seed_end_pos', len(seeds))]
    else:
        seeds_to_run = seeds

    try:
        for seed in seeds_to_run:
            rnd.seed(seed)
            np.random.seed(seed)
            config_to_use = deepcopy(config)
            config_to_use['run_all_seeds'] = False
            # config['seed_pos'] = int(np.argwhere(np.array(seeds_to_run) == seed))
            config_to_use['seed'] = seed
            for dataset in datasets:
                print(dataset)
                # to continue even if there was a problem in a dataset
                main(config_to_use.copy(), dataset, seed)
    except KeyboardInterrupt as e:
        print(e)


if __name__ == '__main__':
    datasets_default = [
        'MedViT2-nopt',
    ]

    if sys.argv[1].endswith('json'):
        # received a json file as config
        run_config(sys.argv[1], datasets_default)
    else:
        # received a folder -> run all configs inside
        for config_file in sorted(os.listdir(sys.argv[1])):
            if config_file.endswith('.json'):
                run_config(os.path.join(sys.argv[1], config_file), datasets_default)
    
