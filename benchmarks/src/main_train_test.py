import sys
import os
import json
from copy import deepcopy
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from utils import *
from edca.encoder import NpEncoder
from edca.utils import class_distribution_distance

DEFAULT_SAVE_DIR = '.'
DEFAULT_DATASETS_SRC_DIR = '../datasets'

# attempt to import served dependent variables
try:
    from config_variables import SAVE_DIR, DATASETS_SRC_DIR
except:
    SAVE_DIR = DEFAULT_SAVE_DIR

def update_config_params(config):
    """update old config type to new"""
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

def main(config, dataset_name, split_seed, framework_seed):
    dataset = os.path.join(DATASETS_SRC_DIR, dataset_name)
    df = pd.read_csv(dataset)
    y = df.pop('class')

    experiment_path = os.path.join(SAVE_DIR, config['save_path'], dataset_name.split('.')[0], f'exp_{datetime.now()}')
    os.makedirs(experiment_path, exist_ok=True)

    config['dataset'] = dataset
    if config.get('fairness_params', None):
        config['fairness_params'] = config['fairness_params'][dataset_name]

    # save config
    with open(os.path.join(experiment_path, 'config.json'), 'w') as file:
        json.dump(config, file, indent=3, cls=NpEncoder)
        file.close()

    # evaluate the frameworks
    results = {}
    results['dataset'] = dataset

    X_train, X_test, y_train, y_test = train_test_split(df, y, 
        test_size=config.get('test_size', 0.3), 
        shuffle=True,
        stratify=y,
        random_state=split_seed)
    results['train_data'] = results.get('train_data', []) + [X_train.shape]
    results['test_data'] = results.get('test_data', []) + [X_test.shape]
    results['dataset_cdd'] = results.get('dataset_cdd', []) + [class_distribution_distance(np.array(y.value_counts(normalize=True)), y.nunique())]
    results['train_cdd'] = results.get('original_train_cdd', []) + [class_distribution_distance(np.array(y_train.value_counts(normalize=True)), y.nunique())]

    train_models(
        results,
        X_train,
        y_train,
        X_test,
        y_test,
        config,
        experiment_path,
        fold=0,
        seed=framework_seed
    )

def run_config(config_path):
    print('>>>>>>>>> Running config:', config_path)

    # read config
    with open(config_path, 'r') as file:
        config = json.load(file)
    config = update_config_params(config)

    if 'dataset' not in config:
        print('No datasets to run')
        return
    
    datasets = config['dataset']
    # (seeds used the main.py)
    framework_seeds = [42, 384, 518, 522, 396, 400, 23, 791, 666, 283, 28, 298, 557, 309, 822, 569, 825, 185, 574, 325, 844, 90, 219, 864, 872, 618, 747, 365, 237, 767]
    # seeds for the splits - total different from the previous ones
    split_seeds = [983, 871, 854, 537, 885, 686, 593, 122, 424, 130, 455, 703, 313, 792, 339, 937, 516, 476, 248, 748, 43, 236, 509, 9, 913, 978, 741, 11, 236, 705]

    start_pos = config.get('seed_pos', 0)
    end_pos = config.get('seed_end_pos', 30)

    if config.get('seed', None):
        framework_seeds_to_run = [config['seed']]
        split_seeds_to_run = [config['seed']]
    elif config.get('run_all_seeds', False) == False:
        framework_seeds_to_run = [framework_seeds[start_pos]]
        split_seeds_to_run = [split_seeds[start_pos]]
    elif config.get('run_all_seeds', False):
        framework_seeds_to_run = framework_seeds[start_pos:end_pos]
        split_seeds_to_run = split_seeds[start_pos:end_pos]
    else:
        framework_seeds_to_run = framework_seeds
        split_seeds_to_run = split_seeds

    try:
        for split_seed in split_seeds_to_run:
            for dataset in datasets:
                for framework_seed in framework_seeds_to_run:
                    print(dataset)
                    config_to_use = deepcopy(config)
                    config_to_use['run_all_seeds'] = False
                    config_to_use['seed'] = framework_seed
                    config_to_use['split_seed'] = split_seed
                    main(config_to_use.copy(), dataset, split_seed, framework_seed)
    except KeyboardInterrupt as e:
        print(e)
    
if __name__ == '__main__':
    if sys.argv[1].endswith('.json'):
        # received a json file as config
        run_config(sys.argv[1])
    else:
        # received a folder -> run all configs inside
        for config_file in sorted(os.listdir(sys.argv[1])):
            if config_file.endswith('.json'):
                run_config(os.path.join(sys.argv[1], config_file))