"""
Script to analyse the selected features by the end pipelines in all the experiments in EDCA
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
from utils import best_individuals_overall
from visuals import plot_bar_comparison
from parameters import datasets, frameworks, fairness_parameters, images_dir, experimentation_name, frameworks_palette
import seaborn as sns


# %%
def calculate_dr_occurrence(individuals, normalize=False):
    dr_type = {
        'Only IS' : 0,
        'Only FS' : 0,
        'IS and FS' : 0,
        'No DR' : 0
    }
    for individual in individuals:
        if 'sample' in individual and 'features' not in individual:
            dr_type['Only IS'] += 1
        elif 'sample' not in individual and 'features' in individual:
            dr_type['Only FS'] += 1
        elif 'sample' in individual and 'features' in individual:
            dr_type['IS and FS'] += 1
        else:
            dr_type['No DR'] += 1
    
    if normalize:
        total = sum(dr_type.values())
        dr_type = {key: value/total for key, value in dr_type.items()}
    return dr_type

    
# %%
results = {}
for dataset in datasets:
    dr_per_framework = {}
    for framework, framework_path in frameworks.items():
        best_individuals = best_individuals_overall(f'{framework_path}/{dataset}')
        dr_occurrence = calculate_dr_occurrence(best_individuals, normalize=True)
        dr_per_framework[framework] = dr_occurrence
        # plot_bar_comparison(dr_occurrence, title=f'{dataset} DR OCC')
    results[dataset] = dr_per_framework

    
# %%
# fig, axs = plt.subplots(ncols=len(datasets), figsize=(len(datasets)*10, 10))
# for i, dataset in enumerate(sorted(results.keys())):
#     plot_bar_comparison(results[dataset], title=dataset, highlight_y=fairness_parameters[f'{dataset}.csv']['sensitive_attributes'], ax=axs[i])
# plt.tight_layout()
# plt.savefig(f'{images_dir}/{experimentation_name}_dr_occurrence_distribution.pdf', format='pdf')

# %%
fig, axs = plt.subplots(ncols=len(datasets),figsize=(len(datasets)*7, 5))
handles, labels = None, None

for i, dataset in enumerate(sorted(results.keys())):
    plot_bar_comparison(
        results[dataset],
        title=dataset,
        highlight_y=fairness_parameters[f'{dataset}.csv']['sensitive_attributes'],
        ax=axs[i],
        palette=frameworks_palette
    )
    axs[i].legend_.remove()
    if handles is None:
        h, l = axs[i].get_legend_handles_labels()
        if h and l:
            handles, labels = h, l


if handles is not None:
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.085))
plt.tight_layout()
plt.savefig(f'{images_dir}/{experimentation_name}_dr_occurrence_distribution.pdf', format='pdf')


# %%
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
def calculate_proportions(data, fairness_params):
    proportions = {}
    if fairness_params['bin_class']:
        for key, encodings in fairness_params['bin_class'].items():
            # print(encodings)
            if key in data.columns:
                data[key] = encode_numeric_feature(data[key], encodings)
    for sensitive_attribute in fairness_params['sensitive_attributes']:
        if sensitive_attribute not in data.columns:
            continue
        
        counts = data[sensitive_attribute].value_counts(normalize=True)
        # counts = counts.add_prefix(f'{get_attribute_name(sensitive_attribute)}_')
        # proportions.update(counts.min().to_dict())
        proportions[get_attribute_name(sensitive_attribute)] = counts.max()
    return proportions


# %% * analyse class proportions
import os
import json

sensitive_proportions = {}
for dataset in datasets:
    values = []
    for framework, path in frameworks.items():
        for exp in os.listdir(os.path.join(path, dataset)):
            if not exp.startswith('exp'):
                continue
            with open(os.path.join(path, dataset, exp, 'config.json')) as file:
                config = json.load(file)
            for fold in range(1, 5+1):
                selected_data = pd.read_csv(os.path.join(path, dataset, exp, 'evo', f'evo_fold{fold}', 'best_data.csv'))
                props = calculate_proportions(selected_data, config['fairness_params'])
                props['framework'] = framework
                values.append(props)
    # calculate original proportions
    data = pd.read_csv(f'../data_amlb/{dataset}.csv')
    props = calculate_proportions(data, config['fairness_params'])
    props['framework'] = 'Original'
    values.append(props)
    sensitive_proportions[dataset] = pd.DataFrame(values)

# %%
for dataset in datasets:
    print(dataset)
    print(sensitive_proportions[dataset].groupby('framework').mean(numeric_only=True))

# %%
values = []
for dataset in datasets:
    mean = sensitive_proportions[dataset].groupby('framework').mean(numeric_only=True)
    std = sensitive_proportions[dataset].groupby('framework').std(numeric_only=True)
    aux = mean.round(6).astype(str) + "±" + std.round(6).astype(str)
    # aux.reset_index(inplace=True, drop=False)
    values.append(aux)
df = pd.concat(values, keys=datasets, names=['dataset'])
df.reset_index(inplace=True, drop=False)

# %%
latex_format = df.to_latex(
    bold_rows=True,
    index=False,
    column_format="|".join(["c"] * len(df.columns))
)

print(latex_format.replace('±nan', '').replace('NaN', '-'))