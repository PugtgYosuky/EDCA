# %%
import pandas as pd
import os
import json
from sklearn import metrics
import fairlearn.metrics as fair_metrics
from utils import get_results_dataframe
import seaborn as sns
import matplotlib.pyplot as plt
from parameters import datasets, frameworks

# %% 
import warnings
warnings.filterwarnings("ignore")

# %%
import sys
sys.path.append('../')
from src.edca.utils import abroca_metric


# %%
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


# %%
def calculate_fairness(file, fairness_params):
    df = pd.read_csv(file)
    # encode target
    df['y_test'] = df['y_test'] == fairness_params['positive_class']
    df['y_pred'] = df['y_pred'] == fairness_params['positive_class']
    
    # encode numeric sensitive features
    if fairness_params['bin_class']:
        for key, encodings in fairness_params['bin_class'].items():
            df[key] = encode_numeric_feature(df[key], encodings)
    # calculate metrics
    tn, fp, fn, tp = metrics.confusion_matrix(df.y_test, df.y_pred).ravel()
    search_metric = 1 - ((metrics.matthews_corrcoef(df.y_test, df.y_pred)+1) / 2)
    fairness_metric = fair_metrics.demographic_parity_difference(df.y_test, df.y_pred, sensitive_features=df[fairness_params['sensitive_attributes']]) + fair_metrics.equalized_odds_difference(df.y_test, df.y_pred, sensitive_features=df[fairness_params['sensitive_attributes']]) + abroca_metric(df[fairness_params['sensitive_attributes']], df.y_test, df.y_proba_1)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    # individual fairness
    values = {}
    for sensitive_attribute in fairness_params['sensitive_attributes']:
        values.update(
        {
            f'{sensitive_attribute}_demographic_parity' : fair_metrics.demographic_parity_difference(df.y_test, df.y_pred, sensitive_features=df[sensitive_attribute]),
            f'{sensitive_attribute}_equalized_odds' : fair_metrics.equalized_odds_difference(df.y_test, df.y_pred, sensitive_features=df[sensitive_attribute]),
            f'{sensitive_attribute}_equal_opportunity' : fair_metrics.equal_opportunity_difference(df.y_test, df.y_pred, sensitive_features=df[sensitive_attribute]),
            f'{sensitive_attribute}_abroca' : abroca_metric(df[sensitive_attribute], df.y_test, df.y_proba_1),
        }
    )
    values.update( {
        'demographic_parity' : fair_metrics.demographic_parity_difference(df.y_test, df.y_pred, sensitive_features=df[fairness_params['sensitive_attributes']]),
        'equalized_odds' : fair_metrics.equalized_odds_difference(df.y_test, df.y_pred, sensitive_features=df[fairness_params['sensitive_attributes']]),
        'equal_opportunity' : fair_metrics.equal_opportunity_difference(df.y_test, df.y_pred, sensitive_features=df[fairness_params['sensitive_attributes']]),
        'abroca' : abroca_metric(df[fairness_params['sensitive_attributes']], df.y_test, df.y_proba_1),
        'roc_auc' : metrics.roc_auc_score(df.y_test, df.y_proba_1),
        'f1' : metrics.f1_score(df.y_test, df.y_pred),
        'mcc' : metrics.matthews_corrcoef(df.y_test, df.y_pred),
        'recall' : metrics.recall_score(df.y_test, df.y_pred),
        'precision' : metrics.precision_score(df.y_test, df.y_pred),
        'accuracy' : metrics.accuracy_score(df.y_test, df.y_pred),
        'balanced acc' : metrics.balanced_accuracy_score(df.y_test, df.y_pred),
        'tpr' : tpr,
        'fpr' : fpr,
        'tp' : tp,
        'tn' : tn,
        'fp' : fp,
        'fn' : fn,
        'cf baseline' : search_metric,
        'cf fairaware-50-50' : 0.5*search_metric + 0.5*(fairness_metric/3),
        'cf fairaware-80-20' : 0.8*search_metric + 0.2*(fairness_metric/3)
    })
    return values
    

# %% 
def calculate_fairness_metrics(exp_path, dataset, setup='evo'):
    # load config
    with open(f'{exp_path}/config.json') as file:
        config = json.load(file)
    # load the predictions and calculate metrics

    predictions_path = os.path.join(exp_path, 'evo', 'predictions')
    prediction_files = sorted([file for file in os.listdir(predictions_path) if file.startswith(f'{setup}_predictions')])

    data = []
    for file in prediction_files:
        data.append(calculate_fairness(os.path.join(predictions_path, file), config['fairness_params']))
    return data


# %%
def calculate(experiment, dataset):
    data = []
    experiments = [ exp for exp in sorted(os.listdir(experiment)) if exp.startswith('exp')]
    for i, exp in enumerate(experiments):
        data += calculate_fairness_metrics(os.path.join(experiment, exp), dataset)
    return data

def calculate_mean(data, name, show_std=True):
    df = pd.DataFrame(data).mean()
    if show_std:
        std_df = pd.DataFrame(data).std()
        df = df.round(2).astype(str) + " ± " + std_df.round(2).astype(str)
    df.name = name
    return pd.DataFrame(df)

def calculate_max(data, name):
    df = pd.DataFrame(data).max()
    df.name = name
    return pd.DataFrame(df)

def calculate_min(data, name):
    df = pd.DataFrame(data).min()
    df.name = name
    return pd.DataFrame(df)

def calculate_median(data, name):
    df = pd.DataFrame(data).median()
    df.name = name
    return pd.DataFrame(df)

# %%
def transform_data(data):
    """receives a list of dicts with the values"""
    print(data[0])
    keys = list(data[0].keys())
    all_values = {key : [] for key in keys}
    for values in data:
        for key, item in values.items():
            all_values[key].append(item)
    return all_values

# %%
def get_results_df(datasets, frameworks):
    values = {
        'max' : [],
        'median' : [],
        'mean' : [],
        'min' : [],
    }
    all_values = {}
    for dataset in datasets:
        for key, experiment in frameworks.items():
            try:
                if key not in all_values:
                    all_values[key] = {dataset:[]}
                if dataset not in all_values[key]:
                    all_values[key][dataset] = []
                print(key)
                data = calculate(f'{experiment}/{dataset}', dataset)
                all_values[key][dataset].append(data)
                avg_res = calculate_mean(data, key, show_std=True)
                median_res = calculate_median(data, key)
                max_res = calculate_max(data, key)
                min_res = calculate_min(data, key)
            except Exception as e:
                print(e)
                continue
            avg_res.columns = pd.MultiIndex.from_product([[dataset], avg_res.columns])
            max_res.columns = pd.MultiIndex.from_product([[dataset], max_res.columns])
            min_res.columns = pd.MultiIndex.from_product([[dataset], min_res.columns])
            median_res.columns = pd.MultiIndex.from_product([[dataset], median_res.columns])
            values['max'].append(max_res)
            values['min'].append(min_res)
            values['mean'].append(avg_res)
            values['median'].append(median_res)

    for key, dfs in values.items():
        results = pd.concat(dfs, axis=1)
        results = results.reindex(sorted(results.columns), axis=1)
        results.to_csv(f'results_{key}.csv', index=True)
        values[key] = results

    return all_values, values

# %%
all_values, dfs = get_results_df(datasets, frameworks)

# %%

frame_results = {}
for dataset in datasets:
    values = []
    for framework in frameworks:
        df = pd.DataFrame(all_values[framework][dataset][0])
        df['framework'] = framework
        values.append(df)
    frame_results[dataset] = pd.concat(values, ignore_index=True)

# %% 
def boxplot_ax(data, x_variable, y_variable, ax, title=''):
    median = data.groupby(y_variable).median()[x_variable].max()
    print(median)
    sns.boxplot(
        data=data,
        x=x_variable,
        y=y_variable,
        width=0.75,
        dodge=True,
        orient='h',
        ax=ax,
        color='grey'
    )
    ax.axvline(x=median, color='red', linestyle='--', linewidth=2, label='Max Median')
    ax.set_title(title, fontdict={'weight':'bold'})
    ax.set_ylabel(y_variable.replace('_', ' '), fontdict={'weight':'bold'})
    ax.set_xlabel(x_variable.replace('_', ' '), fontdict={'weight':'bold'})
# %% 
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
for i, dataset in enumerate(datasets):
    boxplot_ax(frame_results[dataset], 'demographic_parity', 'framework', ax=axs[i//2, i%2], title=dataset)
plt.tight_layout()