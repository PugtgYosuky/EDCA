"""
Script to analyse the selected features by the end pipelines in all the experiments in EDCA
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
from utils import best_individuals_overall
from visuals import plot_bar_comparison
from parameters import datasets, frameworks, fairness_parameters, images_dir, experimentation_name, frameworks_palette, LOGS_ROOT
import seaborn as sns
import os
import glob
import json
import numpy as np
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


def load_reduction_percents_from_run(run_dir, framework="edca"):
    """
    run_dir: .../exp3_mlp/exp_...
    returns dict with sample_%, features_%, data_% for the chosen framework
    """
    fp = os.path.join(run_dir, "results.json")
    if not os.path.isfile(fp):
        return None

    with open(fp, "r") as f:
        r = json.load(f)

    info = r.get("run_info", {}).get(framework, {})
    if not info:
        return None

    return {
        "sample_%": info.get("sample_%", np.nan),
        "features_%": info.get("features_%", np.nan),
        "data_%": info.get("data_%", np.nan),
    }
    
# %%
LOGS_ROOT = os.path.join("..", "logs", "MedViT2-nopt")

results = {}
for dataset in datasets:
    dr_per_framework = {}
    exp_dir = os.path.join(LOGS_ROOT, dataset, experimentation_name)

   

    for framework, framework_folder in frameworks.items():
        all_best = []

        framework_roots = sorted([
            os.path.join(exp_dir, run_dir, framework_folder)
            for run_dir in os.listdir(exp_dir)
            if run_dir.startswith("exp_")
            and os.path.isdir(os.path.join(exp_dir, run_dir, framework_folder))
        ])
       

        for framework_root in framework_roots:
            all_best.extend(best_individuals_overall(framework_root))

        if not all_best:
            raise FileNotFoundError(
                f"No valid runs found for {dataset} / {framework} under {exp_dir} "
                f"(looked for exp_*/{framework_folder})"
            )

        dr_per_framework[framework] = calculate_dr_occurrence(all_best, normalize=True)

    results[dataset] = dr_per_framework

reduction_stats = {}  # per dataset dataframe of runs

for dataset in datasets:
    exp_dir = os.path.join(LOGS_ROOT, dataset, experimentation_name)
    run_dirs = sorted([
        os.path.join(exp_dir, d)
        for d in os.listdir(exp_dir)
        if d.startswith("exp_") and os.path.isdir(os.path.join(exp_dir, d))
    ])

    rows = []
    for run_dir in run_dirs:
        vals = load_reduction_percents_from_run(run_dir, framework="edca")
        if vals is None:
            continue
        rows.append(vals)

    reduction_stats[dataset] = pd.DataFrame(rows)

# Make mean±std tables (one value per dataset)
def mean_pm_std(series, nd=2):
    m = np.nanmean(series.values)
    s = np.nanstd(series.values, ddof=1)
    return f"{m:.{nd}f}±{s:.{nd}f}"

table_data = []
table_instances = []
table_features = []

for dataset in datasets:
    df = reduction_stats[dataset]
    table_data.append({"Dataset": dataset, "EDCA": mean_pm_std(df["data_%"])})
    table_instances.append({"Dataset": dataset, "EDCA": mean_pm_std(df["sample_%"])})
    table_features.append({"Dataset": dataset, "EDCA": mean_pm_std(df["features_%"])})

df_data = pd.DataFrame(table_data)
df_instances = pd.DataFrame(table_instances)
df_features = pd.DataFrame(table_features)

print("\n(a) Percentage of data")
print(df_data.to_latex(index=False))

print("\n(b) Percentage of instances")
print(df_instances.to_latex(index=False))

print("\n(c) Percentage of features")
print(df_features.to_latex(index=False))

fig, axs = plt.subplots(ncols=len(datasets), figsize=(len(datasets)*10, 10))

for i, dataset in enumerate(sorted(results.keys())):
    plot_bar_comparison(
        results[dataset],
        title=dataset,
        highlight_y=fairness_parameters.get(f'{dataset}.csv', {}).get('sensitive_attributes', []),
        ax=axs[i]
    )

plt.tight_layout()
os.makedirs(images_dir, exist_ok=True) 
plt.savefig(f'{images_dir}/{experimentation_name}_dr_occurrence_distribution.pdf', format='pdf')

'''# %%
fig, axs = plt.subplots(ncols=len(datasets),figsize=(len(datasets)*7, 5))
handles, labels = None, None

for i, dataset in enumerate(sorted(results.keys())):
    plot_bar_comparison(
        results[dataset],
        title=dataset,
        #highlight_y=fairness_parameters[f'{dataset}.csv']['sensitive_attributes'],
        highlight_y=fairness_parameters.get(f'{dataset}.csv', {}).get('sensitive_attributes', []),
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
'''