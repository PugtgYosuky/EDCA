# %%
import pandas as pd
import numpy as np
import os
import json
from utils import get_results_dataframe, get_results_df, get_estimation_data
from visuals import plot_estimation_individuals_evolution, plot_individuals_estimation_distribution, boxplot_ax, plot_time_distribution_epms
from stats import statistical_test_repeated
import matplotlib.pyplot as plt
from parameters import datasets, frameworks, images_dir, experimentation_name
from tqdm import tqdm
import pickle
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
# %% 
import warnings
warnings.filterwarnings("ignore")
save_path = None


# %%
# df_by_framework_dataset, df_by_dataset = get_results_df(datasets, frameworks, experimentation_name=experimentation_name, use_checkpoints=True)

# # %% 
# metric = 'f1'
# fig, axs = plt.subplots(nrows=len(datasets), figsize=(12, len(datasets) * 4))
# for i, dataset in enumerate(datasets):
#     boxplot_ax(df_by_dataset[dataset], metric, 'framework', ax=axs[i], title=dataset)
# plt.tight_layout()
# plt.savefig(f'{images_dir}/boxplot_{metric}.pdf', format='pdf')

# # %%
# metric = 'num_evaluated_individuals'
# fig, axs = plt.subplots(nrows=len(datasets), figsize=(12, len(datasets) * 4))
# for i, dataset in enumerate(datasets):
#     boxplot_ax(df_by_dataset[dataset], metric, 'framework', ax=axs[i], title=dataset)
# plt.tight_layout()
# plt.savefig(f'{images_dir}/boxplot_{metric}.pdf', format='pdf')

# %% 
dataset = 'Australian'

# %%
# * evolution of the individuals trained and estimated
dynamic_frameworks = {setup: path for setup, path in frameworks.items() if 'static' not in path and 'Baseline' not in setup}

# %%
for framework, path in dynamic_frameworks.items():
    plot_estimation_individuals_evolution(framework, f'{path}/{dataset}', save_path=None)
    plot_individuals_estimation_distribution(framework, f'{path}/{dataset}', save_path=None)

# %%
# * evolution of the error
for setup, path in frameworks.items():
    if setup == 'Baseline':
        continue
    data = {}
    for exp in os.listdir(f'{path}/{dataset}'):
        try:
            with open(f'{path}/{dataset}/{exp}/edca/edca_fold3/estimators_info.json') as file:
                info = json.load(file)
            for model, values in info.items():
                aux_data, initial_trained = get_estimation_data(values.copy())
                assessment_error = pd.json_normalize(aux_data['assessment_error']).add_prefix('assessment_error_')
                aux_data = pd.concat([aux_data.drop(columns=['assessment_error']), assessment_error], axis=1)
                data[model] = data.get(model, []) + [aux_data]
        except Exception as e:
            print(e)
            continue

    fig, ax = plt.subplots(nrows=len(data), figsize=(15, 5*len(data)))
    fig.suptitle(setup.replace('-', ' ').upper(), fontsize=25, fontweight='bold', y=1.001)
    for idx, (model, values) in enumerate(data.items()):
        df = pd.concat(values)
        df_grouped = df.groupby('generation').mean(numeric_only=True)
        df_grouped = df_grouped.reset_index()
        ax[idx].scatter(df_grouped.generation, df_grouped.assessment_error_mae, alpha=0.3)
        mean_mae = df_grouped.assessment_error_mae.mean()
        # calculate linear fit
        aux = df_grouped.loc[(df_grouped.generation.duplicated() == False) & (df_grouped.generation.isna() == False) & (df_grouped.assessment_error_mae.isna() == False)]
        m, b = np.polyfit(aux.generation, aux.assessment_error_mae, 1)
        
        ax[idx].plot(df_grouped.generation, m*df_grouped.generation + b, color='red', linestyle='--', label=f'{m}x + {b}')
            # ax[idx].axhline(y=mean_mae, linestyle='--', label=f' Avg MAE: {mean_mae:.2f}', color=colours[i])
        # ax[idx].axvline(x=, color='red', linestyle='--', label='Start Estimation')
        ax[idx].set_xlabel('Generation')
        ax[idx].set_ylabel('MAE')
        ax[idx].set_title(f'{model}')
        ax[idx].legend()
        ax[idx].set_ylim([0, 1])
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/estimation-assessment-error-{setup}-mae.png', bbox_inches='tight')
        plt.savefig(f'{save_path}/estimation-assessment-error-{setup}-mae.pdf', bbox_inches='tight')

# %% 

# ** predictions vs estimated
def get_predictions(values):
    y_real = []
    y_estimated = []
    for gen in values:
        if 'y_real' not in gen['assessment_error']:
            continue
        y_real += gen['assessment_error']['y_real']
        y_estimated += gen['assessment_error']['y_estimated']
    return y_real, y_estimated

for setup, path in frameworks.items():
    data = {}
    if setup == 'Baseline':
        continue
    for exp in os.listdir(f'{path}/{dataset}'):
        try:
            with open(f'{path}/{dataset}/{exp}/edca/edca_fold3/estimators_info.json') as file:
                info = json.load(file)
            for model, values in info.items():
                y_real, y_estimated = get_predictions(values)
                if model not in data:
                    data[model] = {'y_real': [], 'y_estimated': []}
                data[model]['y_real'] += y_real
                data[model]['y_estimated'] += y_estimated
        except Exception as e:
            print(e)
            continue
    
    fig, ax = plt.subplots(nrows=len(data), figsize=(6, 6*len(data)))
    fig.suptitle(setup.replace('-', ' ').upper(), fontsize=20, fontweight='bold', y=1.001)
    for idx, (model, values) in enumerate(data.items()):
        ax[idx].scatter(values['y_estimated'], values['y_real'], alpha=0.2, s=2)
        ax[idx].plot([0, 1], [0, 1], 'r--')  # Diagonal line
        ax[idx].set_xlabel('Estimated Values', fontdict={'size':13})
        ax[idx].set_ylabel('True Values', fontdict={'size':13})
        nunique = pd.Series(values['y_estimated']).nunique()
        # ax[idx].set_xlim([0.25, 0.55])
        # calculate linear fit
        if len(values['y_estimated']) > 1:
            m, b = np.polyfit(values['y_estimated'], values['y_real'], 1)
        else:
            m, b = None, None
        mae = mean_absolute_error(values['y_real'], values['y_estimated'])
        r2 = r2_score(values['y_real'], values['y_estimated'])
        ax[idx].set_title(f'{model} \n MAE = {mae:.2f} Corr {np.corrcoef(values["y_real"], values["y_estimated"])[0, 1]:.2f} [{m:.2f}x + {b:.2f}]', fontdict={'size':16})
    plt.tight_layout()

# %%
# ** time distribution analysis
for framework, path in frameworks.items():
    if framework == 'Baseline':
        continue
    plot_time_distribution_epms(framework, f'{path}/{dataset}', save_path=save_path)


# %%
# ** analyse percentage retrain
