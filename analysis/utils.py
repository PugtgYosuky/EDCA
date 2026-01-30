import pandas as pd
import json
import os
import ast
import numpy as np
from scipy import stats
from tqdm import tqdm
from sklearn import metrics
import fairlearn.metrics as fair_metrics
import pickle

import sys
sys.path.append('../')
from src.edca.utils import abroca_metric

import sys
sys.path.append('../src')
from edca.utils import class_distribution_distance

def calculate_mean(data, name, show_std=True):
    df = pd.DataFrame(data).mean(numeric_only=True)
    if show_std:
        std_df = pd.DataFrame(data).std(numeric_only=True)
        df = df.round(2).astype(str) + " ± " + std_df.round(2).astype(str)
    df.name = name
    return pd.DataFrame(df)

def calculate_max(data, name):
    df = pd.DataFrame(data).max(numeric_only=True)
    df.name = name
    return pd.DataFrame(df)

def calculate_min(data, name):
    df = pd.DataFrame(data).min(numeric_only=True)
    df.name = name
    return pd.DataFrame(df)

def calculate_median(data, name):
    df = pd.DataFrame(data).median(numeric_only=True)
    df.name = name
    return pd.DataFrame(df)

def get_results_dataframe(exp_path):
    """  Converts the results json to dataframe"""
    try:
        with open(os.path.join(exp_path, 'results.json'), 'r') as f:
            results = json.load(f)
            results.pop('dataset')
            results.pop('train_data')
            results.pop('test_data')
            results.pop('train_cdd')
            df = pd.DataFrame(results)
            return df
    except:
        return pd.DataFrame()

# def get_experiment_results(experiment_path):
#     """
#     Merges the results from all experiments into a dataframe. 
#     """
#     runs = os.listdir(experiment_path)
#     results = []
#     for run in runs:
#         results.append(get_results_dataframe(os.path.join(experiment_path, run)))
#     return pd.concat(results)


def map_flaml_models(model):
    if model == 'lgbm':
        return 'LGBMClassifier'
    elif model == 'rf':
        return 'RandomForestClassifier'
    elif model == 'xgb_limitdepth' or model == 'xgboost':
        return 'XGBClassifier'
    elif model == 'extra_tree':
        return 'ExtraTreesClassifier'
    elif model == 'lrl2' or model == 'lrl1':
        return 'LogisticRegression'
    else:
        return model
    
def get_classifier(config):
    if isinstance(config, str):
        config = json.loads(config)
    try:
        if 'model' in config:
            return list(config['model'].keys())[0]
        elif 'flaml_estimator' in config:
            return map_flaml_models(config['flaml_estimator'])
    except Exception as e:
        return 'None'
    
def get_class_proportion(class_proportion):
    lst = ast.literal_eval(class_proportion)
    for (a, b) in lst:
        if a == 1:
            return b
    return None


def get_results_values(datasets, frameworks, metric, source_path, symbol='±', round_digits=2):
    values = []
    for dataset in datasets:
        aux = {}
        for framework in frameworks:
            if 'all' in framework:
                framework = framework.replace('-all', '')
                train_data = 'all_data_'
            else:
                train_data = ''
            framework_name = framework
            if 'edca' in framework:
                framework_name = 'evo'

            if 'flaml' in framework and 'edca' in framework:
                framework_name = 'flaml_with_edca'

            if 'tpot' in framework and 'edca' in framework:
                framework_name = 'tpot_with_edca'
            metric_name = metric
            if framework == 'tpot':
                train_data = ''
            if metric_name == 'num_pipelines_tested' and framework == 'flaml':
                metric_name = 'num_iterations'
            if metric_name == 'num_pipelines_tested' and framework == 'tpot':
                metric_name = 'total_evaluated'
            try:
                df = pd.read_csv(os.path.join(source_path, dataset, f'{framework}-results.csv'))
                mean = round(df[f'{framework_name}_{train_data}{metric_name}'].mean(), round_digits)
                std = round(df[f'{framework_name}_{train_data}{metric_name}'].std(), round_digits)
                if metric == 'num_pipelines_tested' or metric == 'num_iterations':
                    mean = int(mean)
                    std = int(std)
                if train_data:
                    aux[f'{framework}-all'] = [f'{mean}{symbol}{std}']
                else:
                    aux[framework] = [f'{mean}{symbol}{std}']
            except Exception as e:
                print(e)
                if train_data:
                    aux[f'{framework}-all'] = [np.nan]
                else:
                    aux[framework] = [np.nan]
        values.append(pd.DataFrame(aux))
    df = pd.concat(values)
    df.index = datasets
    return df


def get_stats_test(source_path, df, main, main_metric, other_frameworks, significance_level, symbol, round_digits=2):
    for framework in other_frameworks:
        framework_name = framework
        if 'all' in framework:
            data_var = 'all_data_'
            framework_name = framework.strip('-all')
        else:
            data_var = ''
        for dataset in df.index:
            framework_name = framework
            if 'all' in framework:
                data_var = 'all_data_'
                framework_name = framework.strip('-all')
            try:
                main_df = pd.read_csv(os.path.join(source_path, dataset, f'{main}-results.csv'))
            except Exception as e:
                continue
            if df.loc[dataset, framework] == '-':
                continue
            try:
                other_df = pd.read_csv(os.path.join(source_path, dataset, f'{framework_name}-results.csv'))
            except Exception as e:
                continue
            metric_name = metric
            if metric == 'num_pipelines_tested' and framework == 'flaml':
                metric_name = 'num_iterations'
            elif metric == 'num_pipelines_tested' and 'tpot' in framework:
                metric_name = 'total_evaluated'

            if 'tpot' in framework:
                framework_name = 'tpot'

            aux_data_var = data_var
            if framework == 'tpot':
                aux_data_var = ''
            if 'edca' in framework:
                framework_name = 'evo'

            if 'flaml' in framework and 'edca' in framework:
                framework_name = 'flaml_with_edca'
                aux_data_var = ''

            if 'tpot' in framework and 'edca' in framework:
                framework_name = 'tpot_with_edca'
                aux_data_var = ''
            other_metric = f'{framework_name}_{aux_data_var}{metric_name}'
            a = main_df[main_metric].round(round_digits)
            b = other_df[other_metric].round(round_digits)
            a = a.loc[a.notnull() & b.notnull()]
            b = b.loc[a.notnull() & b.notnull()]
            try:
                st = stats_test(a, b, significance_level=significance_level)
                if st:
                    if st.pvalue <= significance_level:
                        if symbol == None:
                            df.loc[dataset, framework] = f'\\textbf{{{df.loc[dataset, framework]}}}'
                        else:
                            df.loc[dataset, framework] = f'{df.loc[dataset, framework]}{symbol}'
            except:
                continue


def get_experiments_df(src_path, edca_exp=False, number_experiments=None, start_name='edca'):
    """
    Get the results dataframe from the source path for a framework on a specified dataset
    """
    values = []
    exps = sorted(os.listdir(src_path))
    if number_experiments:
        exps = exps[:number_experiments]
    for exp in exps:
        # replace evo with edca on the entire results.json file
        with open(os.path.join(src_path, exp, 'results.json'), 'r') as file:
            content = file.read()

        # Replace all occurrences of "evo" with "edca"
        updated_content = content.replace("evo", "edca")

        # Write the modified content back to a file
        with open(os.path.join(src_path, exp, 'results.json'), 'w') as file:
            file.write(updated_content)
        
        with open(os.path.join(src_path, exp, 'results.json'), 'r') as f:
            data = json.load(f)
        aux = {}
        # get mean results for the different folds
        for key, value in data.items():
            if isinstance(value, list) and (isinstance(value[0], float) or isinstance(value[0], int)):
                aux[key] = [np.nanmean(np.array(value, dtype=np.float32))]

        number_evaluated = []
        for dir in os.listdir(os.path.join(src_path, exp,  'edca')):
            if 'edca_fold' in dir:
                try:
                    individuals_info = pd.read_csv(os.path.join(src_path, exp,  'edca', dir, 'f.csv'))
                    number_evaluated.append(len(individuals_info))
                except:
                    continue
        aux['edca_evaluated_individuals'] = np.nanmean(number_evaluated)

        if edca_exp:
            balance_metrics = {
                'cdd_original': [],
                'cdd_final': [],
                'entropy_original': [],
                'entropy_final': [],
            }
            for fold in range(1, len(data[f'edca_best'])+1):
                try:
                    internal_train_data = pd.read_csv(f'{src_path}/{exp}/{start_name}/{start_name}_fold{fold}/internal_train_data.csv')
                    final_data = pd.read_csv(f'{src_path}/{exp}/{start_name}/{start_name}_fold{fold}/best_data.csv')
                    balance_metrics['cdd_original'].append(class_distribution_distance(internal_train_data.target_class.value_counts(normalize=True).values), internal_train_data.target_class.nunique())
                    balance_metrics['cdd_final'].append(class_distribution_distance(final_data.target_class.value_counts(normalize=True).values), internal_train_data.target_class.nunique())
                    balance_metrics['entropy_original'].append(stats.entropy(internal_train_data.target_class.value_counts(normalize=True).values, base=2))
                    balance_metrics['entropy_final'].append(stats.entropy(final_data.target_class.value_counts(normalize=True).values, base=2))
                except:
                    continue
                
            for key, value in balance_metrics.items():
                aux[key] = np.nanmean(value)
        values.append(pd.DataFrame(aux))
    return pd.concat(values).reset_index()

def get_features_name(dataset_name):
    """
    receives the dataset name and outputs the features names from the original dataset
    """
    dataset = pd.read_csv(f'../data_amlb/{dataset_name}.csv')
    features = list(dataset.columns)
    features.remove('class')
    return features

def calculate_proportion_sensitives(features, sensitive_attributes):
    return len(set(features) & set(sensitive_attributes)) / len(features)

def confusion_elements(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    return TP, FP, FN, TN

def get_attribute_name(attribute):
    attributes_map = {
        'age' : ['age'],
        'race' : ['race'],
        'gender' : ['gender', 'sex', 'x2'],
        'marital' : ['marital', 'x4'],
        'education' : ['x3']
    }
    if attribute in attributes_map.keys():
        return attribute
    for key, values in attributes_map.items():
        if attribute in values:
            return key
        
def encode_numeric_feature(feature_values, encoding_values):
    """ Encodes values of a given numeric feature based on the encoding points
    ! Note: It works only with pandas DataFrames
    """
    # sort encoding for ranges
    encoding_values = list(sorted(encoding_values))
    series = pd.Series([None]*len(feature_values))
    feature_values = feature_values.copy().reset_index(drop=True)
    # encode the limits
    series.loc[feature_values < encoding_values[0]] = f'< {encoding_values[0]}'
    series.loc[feature_values > encoding_values[-1]] = f' > {encoding_values[-1]}'
    
    # encode the mid ranges
    for i in range(len(encoding_values)-1):
        min_value, max_value = encoding_values[i], encoding_values[i+1]
        if len(encoding_values) > 2 and max_value != encoding_values[-1]:
            max_value -= 1
        series.loc[(feature_values >=min_value) & (feature_values <=max_value)] = f'{min_value}-{max_value}'
    return series.tolist()

def calculate_metrics_from_predictions(file, fairness_params):
    """calculates the metrics according to the predictions"""
    df = pd.read_csv(file)
    # calculate metrics
    if len(df.y_test.unique()) == 2:
        tn, fp, fn, tp = metrics.confusion_matrix(df.y_test, df.y_pred).ravel()
    else:
        tn, fp, fn, tp = confusion_elements(df.y_test, df.y_pred)
    search_metric = 1 - ((metrics.matthews_corrcoef(df.y_test, df.y_pred)+1) / 2)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    # individual fairness
    values = {}
    
    if fairness_params: # transform data and calculate fairness metrics
        # encode target
        df['y_test'] = df['y_test'] == fairness_params['positive_class']
        df['y_pred'] = df['y_pred'] == fairness_params['positive_class']
        
        # encode numeric sensitive features
        if fairness_params['bin_class']:
            for key, encodings in fairness_params['bin_class'].items():
                df[key] = encode_numeric_feature(df[key], encodings)
            fairness_metric = fair_metrics.demographic_parity_difference(df.y_test, df.y_pred, sensitive_features=df[fairness_params['sensitive_attributes']]) + fair_metrics.equalized_odds_difference(df.y_test, df.y_pred, sensitive_features=df[fairness_params['sensitive_attributes']]) + abroca_metric(df[fairness_params['sensitive_attributes']], df.y_test, df.y_proba_1)
            for sensitive_attribute in fairness_params['sensitive_attributes']:
                attribute_name = get_attribute_name(sensitive_attribute)
                values.update(
                {
                    f'{attribute_name}_demographic_parity' : fair_metrics.demographic_parity_difference(df.y_test, df.y_pred, sensitive_features=df[sensitive_attribute]),
                    f'{attribute_name}_equalized_odds' : fair_metrics.equalized_odds_difference(df.y_test, df.y_pred, sensitive_features=df[sensitive_attribute]),
                    f'{attribute_name}_equal_opportunity' : fair_metrics.equal_opportunity_difference(df.y_test, df.y_pred, sensitive_features=df[sensitive_attribute]),
                    f'{attribute_name}_abroca' : abroca_metric(df[sensitive_attribute], df.y_test, df.y_proba_1),
                }
            )
            values.update({
            'demographic_parity' : fair_metrics.demographic_parity_difference(df.y_test, df.y_pred, sensitive_features=df[fairness_params['sensitive_attributes']]),
            'equalized_odds' : fair_metrics.equalized_odds_difference(df.y_test, df.y_pred, sensitive_features=df[fairness_params['sensitive_attributes']]),
            'equal_opportunity' : fair_metrics.equal_opportunity_difference(df.y_test, df.y_pred, sensitive_features=df[fairness_params['sensitive_attributes']]),
            'abroca' : abroca_metric(df[fairness_params['sensitive_attributes']], df.y_test, df.y_proba_1)})

    # add performance metrics 
    values.update({
    # 'roc_auc' : metrics.roc_auc_score(df.y_test, df.y_proba_1),
    'f1' : metrics.f1_score(df.y_test, df.y_pred, average='weighted'),
    'mcc' : metrics.matthews_corrcoef(df.y_test, df.y_pred),
    'recall' : metrics.recall_score(df.y_test, df.y_pred, average='weighted'),
    'precision' : metrics.precision_score(df.y_test, df.y_pred, average='weighted'),
    'accuracy' : metrics.accuracy_score(df.y_test, df.y_pred),
    'balanced acc' : metrics.balanced_accuracy_score(df.y_test, df.y_pred),
    'tpr' : tpr,
    'fpr' : fpr,
    'tp' : tp,
    'tn' : tn,
    'fp' : fp,
    'fn' : fn,
    # 'cf baseline' : search_metric,
    # 'cf fairaware-50-50' : 0.5*search_metric + 0.5*(fairness_metric/3),
    # 'cf fairaware-80-20' : 0.8*search_metric + 0.2*(fairness_metric/3)
    })
    return values

def get_experiment_results(run_path, dataset, setup='edca'):
    """
    Get the results dataframe from the source path for a framework on a specified dataset for a specific run
    """

    # replace evo with edca on the entire results.json file
    with open(os.path.join(run_path, 'results.json'), 'r') as file:
        results = json.load(file)

    with open(os.path.join(run_path, 'config.json'), 'r') as file:
        config = json.load(file)

    predictions_path = os.path.join(run_path, setup, 'predictions')
    prediction_files = sorted([file for file in os.listdir(predictions_path) if file.startswith(f'{setup}_predictions')])

    data = []
    nfolds = 0
    for file in prediction_files:
        data.append(calculate_metrics_from_predictions(os.path.join(predictions_path, file), fairness_params=config.get('fairness_params', {})))
        nfolds += 1
    
    ## add other values 
    for key, values in results.items():
        key_name = key.replace(f'{setup}_', '')
        if isinstance(values, list) and len(values) == len(data) and key_name not in data[0].keys():
            nonan_values = [val for val in values if val]
            for i, val in enumerate(nonan_values):
                data[i][key_name] = val
            # add Nones to the end
            if len(nonan_values) != nfolds:
                for i in range(len(nonan_values), nfolds):
                    data[i][key_name] = None
    # add evaluated individuals
    edca_folds_dirs = sorted([folder for folder in os.listdir(os.path.join(run_path, setup)) if folder.startswith(f'{setup}_fold')])
    for i, folder in enumerate(edca_folds_dirs):
        try:
            with open(os.path.join(run_path, setup, folder, 'evaluated_individuals.json')) as file:
                individuals_info = json.load(file)
            data[i]['num_evaluated_individuals'] = len(individuals_info)
        except:
            try:
                individuals_info = pd.read_csv(os.path.join(run_path, setup, folder, 'evaluated_individuals.csv'))
                data[i]['num_evaluated_individuals'] = len(individuals_info)
            except:
                pass

    if config.get('fairness_params', {}):
        # add positive class proportion
        for i, value in enumerate(results[f'{setup}_class_proportions']):
            proportions = json.loads(value)
            positive_class = str(config['fairness_params']['positive_class'])
            data[i]['positive_class_prop'] = round((proportions[positive_class] / sum(list(proportions.values()))) * 100, 2)
        
        # calculate proportion of sensitive attributes over the overall selected data
        sensitive_attributes = config['fairness_params']['sensitive_attributes']
        features_names = get_features_name(dataset)
        for i, best in enumerate(results[f'{setup}_best']):
            if 'features' in best:
                # selected features
                features = [features_names[indice] for indice in best['features']]
            else:
                # use all features
                features = features_names
            data[i]['sensitives_proportion'] = round(calculate_proportion_sensitives(features, sensitive_attributes)*100, 2)
    

    for item in data:
        item['experiment'] = run_path
        item['seed'] = config['seed']

    return data

from collections import defaultdict
def average_dict_results(data):
    sums = defaultdict(float)
    for item in data:
        for key, value in item.items():
            sums[key] += value
    avg_results = {key : value / len(data) for key, value in sums.items()}
    return avg_results


def get_experiment_avg_results(run_path, dataset, setup):
     # replace evo with edca on the entire results.json file
    with open(os.path.join(run_path, 'results.json'), 'r') as file:
        results = json.load(file)

    with open(os.path.join(run_path, 'config.json'), 'r') as file:
        config = json.load(file)

    predictions_path = os.path.join(run_path, setup, 'predictions')
    prediction_files = sorted([file for file in os.listdir(predictions_path) if file.startswith(f'{setup}_predictions')])
    
    data = []
    for file in prediction_files:
        data.append(calculate_metrics_from_predictions(os.path.join(predictions_path, file), fairness_params=config.get('fairness_params', {})))
    
    avg_data = average_dict_results(data)

    # calculate other metrics
    for key, values in results.items():
        key_name = key.replace(f'{setup}_', '')
        if isinstance(values, list) and key_name not in avg_data.keys():
            try:
                avg_data[key_name] = np.nanmean([val for val in values if val])
            except Exception as e:
                continue
    # add evaluated individuals
    edca_folds_dirs = sorted([folder for folder in os.listdir(os.path.join(run_path, setup)) if folder.startswith(f'{setup}_fold')])
    evaluated_individuals = []
    for i, folder in enumerate(edca_folds_dirs):
        try:
            with open(os.path.join(run_path, setup, folder, 'evaluated_individuals.json')) as file:
                individuals_info = json.load(file)
            evaluated_individuals.append(len(individuals_info))
        except:
            try:
                individuals_info = pd.read_csv(os.path.join(run_path, setup, folder, 'evaluated_individuals.csv'))
                evaluated_individuals.append(len(individuals_info))
            except:
                pass
    avg_data['num_evaluated_individuals'] = np.nanmean(evaluated_individuals)
    return [avg_data]


def get_framework_results(experiment, dataset, setup, calculation_type='avg'):
    # calculates the results for each experiment
    data = []
    experiments = [exp for exp in sorted(os.listdir(experiment)) if exp.startswith('exp')]
    for exp in tqdm(experiments):
        try:
            if calculation_type == 'avg':
                # calculate the avg of each run
                data += get_experiment_avg_results(os.path.join(experiment, exp), dataset, setup=setup)
                
            else:
                # calculate all values of all folds
                data += get_experiment_results(os.path.join(experiment, exp), dataset, setup=setup)
        except Exception as e:
            print(e)
            continue
    return data


def get_results_df(datasets, frameworks, experimentation_name, checkpoints_path='', start_name='edca', calculation_type='all'):
    ### !USE THIS to retrieve all the runs for all the experiments and datasets
    
    DF_BY_DATASET = f'{checkpoints_path}/checkpoints/{experimentation_name}_df_by_dataset.pkl'
    DF_BY_FRAMEWORK_DATASET = f'{checkpoints_path}/checkpoints/{experimentation_name}_df_by_framework_dataset.pkl'
    if checkpoints_path and os.path.exists(DF_BY_DATASET):
        # already saved - load them
        with open(DF_BY_DATASET, 'rb') as file:
            df_by_dataset = pickle.load(file)
        with open(DF_BY_FRAMEWORK_DATASET, 'rb') as file:
            df_by_framework_dataset = pickle.load(file)
        return df_by_framework_dataset, df_by_dataset

    df_by_framework_dataset = {}
    for dataset in datasets:
        print('> ', dataset)
        for key, experiment in frameworks.items():
            try:
                if key not in df_by_framework_dataset:
                    df_by_framework_dataset[key] = {}
                df_by_framework_dataset[key][dataset] = get_framework_results(f'{experiment}/{dataset}', dataset, start_name, calculation_type=calculation_type)

            except Exception as e:
                print(e)
                continue

    # merge results by dataset
    df_by_dataset = {}
    for dataset in datasets:
        values = []
        for framework in frameworks:
            df = pd.DataFrame(df_by_framework_dataset[framework][dataset])
            df['framework'] = framework
            values.append(df)
        df_by_dataset[dataset] = pd.concat(values, ignore_index=True)

    # save for later
    if checkpoints_path:
        with open(DF_BY_DATASET, 'wb') as file:
            pickle.dump(df_by_dataset, file)
        with open(DF_BY_FRAMEWORK_DATASET, 'wb') as file:
            pickle.dump(df_by_framework_dataset, file)
    return df_by_framework_dataset, df_by_dataset


def get_individuals_from_experiment(exp_path):
    """
    Receives the experiment path and retrieves the best individuals
    """
    try:
        with open(os.path.join(exp_path, 'results.json')) as file:
            results = json.load(file)
            individuals = [individual for individual in results['evo_best'] if individual]
            return individuals
    except:
        return []
    
def best_individuals_overall(framework_path):
    """
    Outputs all the end pipelines achieved in all the experiments fro the framework path received
    """
    individuals = []
    for exp in sorted(os.listdir(framework_path)):
        individuals += get_individuals_from_experiment(os.path.join(framework_path, exp))
    return individuals


def get_estimation_data(info):
    """ create a dataframe from the estimators info dictionary """
    info = info.copy()
    # remove position=1 at the generation parameter in info
    initial_trained = info.pop(1)
    df = pd.DataFrame(info)
    # df = df.loc[df.generation.duplicated() == False]

    return df, initial_trained


def get_time_data(path):
    """ gets the time information for the specified experiment path"""
    with open(os.path.join(path, 'config.json')) as file:
        config = json.load(file)
    # calculate train geenrations
    train_generations = config['estimation_params'].get('estimation_train_generations', 10) +1 
    # get data
    with open(f'{path}/edca/edca_fold3/optimisation_time_information.json') as file:
        time_results = json.load(file)
    with open(f'{path}/edca/edca_fold3/estimators_info.json') as file:
        estimation_info = json.load(file)

    time_info = {
        'process_estimation' : 0,
        'estimating_individuals' : 0,
        'estimating_overall' : 0,
        'evaluating_individuals' : 0,
        'updating_epm' : 0,
    }
    time_info_models = {}
    # calculate overall times and for each model
    for model, values in estimation_info.items():
        time_info_models[model] = time_info.copy()
        for i, gen_info in enumerate(values):
            # print(model, gen_info['generation'])
            if i == 0:
                time_info_models[model]['initial_train'] = gen_info.get('time_training_epm', 0)
            else:
                for key in time_info.keys():
                    time_info_models[model][key] += gen_info.get(f'time_{key}', 0)
                    time_info[key] += gen_info.get(f'time_{key}', 0)
    # add optimisation time metrics
    time_info['initial_train'] = time_results.get('train_time_overall', 0)
    # get initial evaluation time of the first train_generations
    time_info['initial_evaluation'] = sum(time_results.get('evaluation_time', [0]*train_generations)[:train_generations])
    return time_info, time_info_models

def calculate_average_time(src_path):
    """ calculates the average time information across several experiments in the specified path"""
    time_info = []
    time_info_models = []
    for exp in sorted(os.listdir(src_path)):
        time_info_aux, time_info_models_aux = get_time_data(os.path.join(src_path, exp))
        time_info.append(time_info_aux)
        time_info_models.append(time_info_models_aux)
    avg_time_info = pd.DataFrame(time_info).mean().to_dict()
    # calculate for models
    models = time_info_models[0].keys()
    avg_time_info_models = {}
    for model in models:
        aux = []
        for values in time_info_models:
            aux.append(values[model])
        avg_time_info_models[model] = pd.DataFrame(aux).mean().to_dict()
    
    return avg_time_info, avg_time_info_models


