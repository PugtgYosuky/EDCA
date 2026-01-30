# %%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from parameters import datasets, frameworks, images_dir, experimentation_name
import ast
import pickle
import json
from visuals import *

# %%
def compute_mcc(row):
    TP, TN, FP, FN = row['tp'], row['tn'], row['fp'], row['fn']
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return numerator / denominator if denominator != 0 else 0

def load_dataframe(file, path):
    """ loads the received csv from bests or population results and transforms the proportions per class into features"""
    df = pd.read_csv(file)
    # add metrics
    try:
        df['recall'] = df['tp'] / (df['tp']+df['fn'])
        df['precision'] = df['tp'] / (df['tp']+df['fp'])
        # mcc
        df['mcc'] = df.apply(compute_mcc, axis=1)
    except:
        pass
    proportions = pd.json_normalize(df.proportion_per_class.apply(ast.literal_eval)).add_prefix('proportion_class_')
    with open(os.path.join(path, 'config.json')) as file:
        config = json.load(file)
    if 'fairness_params' in config:
        try:
            positive_class = config['fairness_params']['positive_class']
            df['proportion_class_positive'] = proportions[f'proportion_class_{positive_class}']
        except:
            pass
    return pd.concat([df, proportions], axis=1)

# %%
def get_bests_results_experiments(path):
    values = {}
    for exp in sorted(os.listdir(path)):
        if not exp.startswith('exp'):
            continue
        exp_path = os.path.join(path, exp)
        folds = sorted([folder for folder in os.listdir(os.path.join(exp_path, 'evo')) if folder.startswith('evo_fold')])
        for fold_file in folds:
            try:
                bests = load_dataframe(os.path.join(exp_path, 'evo', fold_file, 'bests.csv'), exp_path)
                for metric in [metric for metric in bests.columns if metric not in ['Iteration', 'config']]:
                    values[metric] = values.get(metric, []) + [list(bests[metric])]
            except Exception as e:
                print(e)
                continue
    return values

# %%
# get bests data
BEST_DATA_STORAGE = f'checkpoints/{experimentation_name}_bests_information.pkl'
if os.path.exists(BEST_DATA_STORAGE):
    with open(BEST_DATA_STORAGE, 'rb') as file:
        data = pickle.load(file)
else:
    data = {}
    for framework, path in frameworks.items():
        if framework not in data:
            data[framework] = {}
        for dataset in datasets:
            # try:
                data[framework][dataset] = get_bests_results_experiments(os.path.join(path, dataset))
            # except:
                # continue
    # save for latter
    with open(BEST_DATA_STORAGE, 'wb') as file:
        pickle.dump(data, file)

# %%
def get_mean_array(array, until_min=False):
    """ calculates the average and std of all the evolution lines"""
    max_generations = max(len(v) for v in array)
    min_generations = min(len(v) for v in array)
    max_exps = len(array)

    data = np.zeros(shape=(max_exps, max_generations))
    data[::] = np.nan
    for i, v in enumerate(array):
        data[i, :len(v)] = v
    
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    if until_min:
        mean = mean[:min_generations]
        std = std[:min_generations]
    return mean, std

def get_label_name(labelname):
    if labelname == 'search_metric':
        label = 'Prediction Error'
    elif labelname == 'recall':
        label = 'TPR'
    elif '_' in labelname:
        label = labelname.replace('_', ' ').capitalize()
    elif labelname == 'recall':
        label = 'Recall'
    elif labelname == 'train_percentage':
        label = 'Data %'
    elif labelname == 'samples_percentage':
        label = 'Instances %'
    elif labelname == 'features_percentage':
        label = 'Features %'
    elif labelname == 'proportion_class_positive':
        label = '% Positive Class'
    else:
        label = labelname.upper()
    if label in ['Recall', 'MCC', 'F1', 'Roc Auc', '% Positive Class', 'PRECISION', 'TPR']:
        label = label + '↑'
    else:
        label = label + '↓'
    return label

# %%
def plot_evolution_metrics(info, metrics, until_min=True, ylim=[0, 1], highlight=[], title='', show_std=False, ax=None):
    if ax:
        graph = ax
    else:
        plt.figure(figsize=(10, 7))
        graph = plt

    cmap = plt.get_cmap('tab20c')
    metrics_colors = {
        'abroca': 0,
        'demographic_parity':1 , 
        'equalized_odds': 2,
        'mcc': 4,
        'tpr': 5
    }
    for metric in metrics:
        lw = 3
        if metric in highlight:
            lw = 5
        avg_metric, std_metric = get_mean_array(info[metric], until_min=until_min)
        generations = np.arange(1, len(avg_metric)+1)
        if show_std:
            graph.fill_between(generations, avg_metric - std_metric, avg_metric + std_metric, alpha=0.2)
        graph.plot(list(range(1, len(avg_metric)+1)), avg_metric, label=get_label_name(metric), lw=lw, color=cmap.colors[metrics_colors[metric]])
    if ax == None:
        plt.xlabel('Generation')
        plt.ylabel('Value')
        plt.ylim(ylim)
        plt.title(title)
    else:
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel('Generation', fontdict={'size':18, 'weight':'bold'})
        ax.set_ylabel('Value', fontdict={'size':18, 'weight':'bold'})
        ax.set_ylim(ylim)
        ax.set_title(title, fontdict={'size':22, 'weight':'bold'})
    graph.legend()


# %%
def plot_evolution_per_metric(data, metrics):
    for framework in data.keys():
        for dataset in data[framework].keys():
            fig, axs = plt.subplots(nrows=len(metrics), figsize=(10, 4*len(metrics)))
            for i, metric in enumerate(metrics):
                avg_metric, _ = get_mean_array(data[framework][dataset][metric], until_min=True)
                axs[i].plot(avg_metric)
                axs[i].set_xlabel('Generation')
                axs[i].set_ylabel(metric)
                axs[i].set_title(f'{framework} [{dataset}]')
                axs[i].set_ylim([0.1, 0.6])
            plt.legend()
            plt.tight_layout()
            plt.show()

# %%
def plot_components_all(data, dataset, metrics, ylim=[0, 1], until_min=False, show_std=False):
    fig, axs = plt.subplots(ncols=len(data.keys()), figsize=(7*len(data.keys()), 4), constrained_layout=True)
    fig.suptitle(dataset, fontweight='bold', fontsize=26, y=0.92)
    fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0, hspace=0)
    handles, labels = None, None
    for i, framework in enumerate(data.keys()):
        ax = axs[i]
        plot_evolution_metrics(data[framework][dataset], metrics=metrics, ylim=ylim, until_min=until_min, title=framework, show_std=show_std, ax=ax)
        axs[i].legend_.remove()
        # Collect legend info from the first subplot that has one
        if handles is None:
            h, l = axs[i].get_legend_handles_labels()
            if h and l:
                handles, labels = h, l
    # fig.subplots_adjust(top=1.5)
    plt.tight_layout()
    fig.savefig(f'{images_dir}/{experimentation_name}_evolution_metrics_{dataset}.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)
    fig, axs = plt.subplots(figsize=(7*len(data.keys()), 0.8))
    axs.axis('off')
    if handles is not None:
        fig.legend(handles, labels, loc='center', ncol=len(handles), fontsize=16)
    fig.savefig(f'{images_dir}/{experimentation_name}_evolution_metrics_{dataset}_legend.pdf', format='pdf')
    plt.show()
    plt.close(fig)
# %% 
def plot_fitness(data, frameworks, datasets):
    for dataset in datasets:
        fig, axs = plt.subplots(nrows=len(frameworks), figsize=(7, 4 * len(frameworks)), sharex=True)
        fig.suptitle(dataset, fontweight='bold', fontsize=32)
        for i, framework in enumerate(frameworks):
            ax = axs[i]
            plt.figure(figsize=(10, 7))
            avg_fit, std_fit = get_mean_array(data[framework][dataset]['fitness'], until_min=True)
            generations = np.arange(1, len(avg_fit)+1)
            ax.fill_between(generations, avg_fit - std_fit, avg_fit + std_fit, alpha=0.2)
            ax.plot(generations, avg_fit, label='Fitness', lw=2)

            avg_avg_fit, std_avg_fit = get_mean_array(data[framework][dataset]['average_fitness'], until_min=True)
            ax.fill_between(generations, avg_avg_fit - std_avg_fit, avg_avg_fit + std_avg_fit, alpha=0.2)
            ax.plot(generations, avg_avg_fit, label='Avg Fitness', lw=2)

            max_val = np.max(get_mean_array(data[framework][dataset]['average_fitness']))
            ax.set_ylim([0.15, 0.45])
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.set_title(f'{framework}')
            ax.legend()
            plt.tight_layout()
        fig.savefig(f'{images_dir}/{experimentation_name}_evolution_fitness_{dataset}.pdf', format='pdf', bbox_inches='tight')
        plt.show()

# %% rename recall to tpr
for framework in data.keys():
    for dataset in data[framework].keys():
        if 'tpr' not in data[framework][dataset]:
            data[framework][dataset]['tpr'] = data[framework][dataset].pop('recall')

# %%
plot_fitness(data, frameworks=frameworks, datasets=datasets)

# %%
plot_evolution_per_metric(data, ['search_metric', 'demographic_parity', 'equalized_odds', 'abroca'])

#%%
ylim = {
    'adult' : [0, 1],
    'credit-card' : [0, 1],
    'portuguese-bank-marketing' : [0.1, 0.8]
}
for i, dataset in enumerate(datasets):
    plot_components_all(data, 
        dataset=dataset, 
        metrics=['mcc', 'tpr', 'abroca', 'demographic_parity', 'equalized_odds'], 
        until_min=True, 
        ylim=ylim.get(dataset, None),
        show_std=False,
)

# %%
def convert_to_single_array(matrix):
    values = []
    for array in matrix:
        values += array
    return values

# %%
def plot_relationship(data, dataset,  x_var, y_vars):
    for framework in data.keys():
        print(framework, dataset)
        plt.figure(figsize=(10, 7))
        if isinstance(data[framework][dataset][x_var][0], list):
            x_points = convert_to_single_array(data[framework][dataset][x_var])
        else:
            x_points = data[framework][dataset][x_var]
        for y_var in y_vars:
            if isinstance(data[framework][dataset][y_var][0], list):
                y_points = convert_to_single_array(data[framework][dataset][y_var])
            else:
                y_points = data[framework][dataset][y_var]
            x_points = np.array(x_points)
            y_points = np.array(y_points)
            indexes = (x_points != np.nan) & (y_points != np.nan)
            correlation = np.round(np.corrcoef(x_points[indexes], y_points[indexes])[0, 1], 3)
            plt.plot(x_points, y_points , '*', label=f'{y_var} [Corr: {correlation}]')
        plt.xlabel(x_var)
        plt.ylabel('Fairness')
        plt.legend()
        plt.title(f'{framework} - {dataset}')
        plt.show()


# %%
for dataset in datasets:
    plot_relationship(data, dataset, 'search_metric', y_vars=['fairness_metric'])



# %% plot populations
def get_last_generation(population_path):
    selected = [pop for pop in os.listdir(population_path) if pop.startswith('Population_generation')]
    values = [value.replace('Population_generation_', '').replace('.csv', '')for value in selected]
    values = [int(value) for value in values if value.isdigit()]
    max_gen = max(values)
    return max_gen

# %%
info = {}
for framework, path in frameworks.items():
    if framework not in info:
        info[framework] = {}
    for dataset in datasets:
        if dataset not in info[framework]:
            info[framework][dataset] = {}
        for exp in sorted(os.listdir(os.path.join(path, dataset))):
            for fold in range(1, 5+1):
                try:
                    max_gen = get_last_generation(os.path.join(path, dataset, exp, 'evo', f'evo_fold{fold}', 'population'))
                    df = load_dataframe(os.path.join(path, dataset, exp, 'evo', f'evo_fold{fold}', 'population', path=f'Population_generation_{max_gen}.csv'))
                    for col in df.columns:
                        info[framework][dataset][col] = info[framework][dataset].get(col, []) + list(df[col])
                except Exception as e:
                    print(e)
                    break

# %%
plot_relationship(info, 'adult', x_var='search_metric', y_vars=['demographic_parity', 'equalized_odds', 'abroca'])



