# %%
import pandas as pd
import numpy as np
import os
import json
from utils import get_results_dataframe, get_results_df
from visuals import boxplot_ax
import numpy as np
from stats import statistical_test_repeated
import matplotlib.pyplot as plt
from parameters import datasets, frameworks, images_dir, experimentation_name, save_path, frameworks_palette
from tqdm import tqdm
import pickle
from scipy import stats

# %% 
import warnings
warnings.filterwarnings("ignore")

# %%
df_by_framework_dataset, df_by_dataset = get_results_df(
    datasets, 
    frameworks, 
    experimentation_name=f'{experimentation_name}_all', 
    start_name='evo', 
    checkpoints_path=save_path,
    calculation_type='all')

# %%
fig, axs = plt.subplots(ncols=len(df_by_dataset.keys()), figsize=(len(datasets)*6, 5))
handles, labels = None, None
markers = {
    'Baseline' : 'o',
    'Fairness-Aware' : 'X'
}
markers_text = {
    'Baseline' : '\u25CF',
    'Fairness-Aware' : '\u00D7'
}
for i, (dataset, df) in enumerate(df_by_dataset.items()):
    aux = df.copy()
    aux['Avg Fairness (DP + EO + ABROCA)'] = aux[['demographic_parity', 'equalized_odds', 'abroca']].mean(axis=1)
    # !look at this
    aux['mcc_scaled'] = (aux['mcc'] + 1) / 2
    aux['Avg Performance (MCC+TPR)'] = aux[['mcc_scaled', 'tpr']].mean(axis=1)
    for ixd_frame, framework in enumerate(sorted(aux.framework.unique())):
        split = aux.loc[aux.framework == framework]
        corr = np.corrcoef(split['Avg Fairness (DP + EO + ABROCA)'], split['Avg Performance (MCC+TPR)'])
        axs[i].plot(split['Avg Performance (MCC+TPR)'], split['Avg Fairness (DP + EO + ABROCA)'], 
                    markers[framework], markersize=8,
                    label=framework.replace('Fair', 'Fairness'), 
                    color=frameworks_palette[framework], alpha=0.7)
        axs[i].text(
            0.1, 0.1 - 0.05*ixd_frame,
            f'{markers_text[framework]} Correlation: {round(corr[0][1], 3)}',
            fontsize=12,
            color=frameworks_palette[framework],
            transform=axs[i].transAxes

        )
    axs[i].set_xlabel('Predictive Performance ↑', fontdict={'size':20, 'weight':'bold'})
    axs[i].set_ylabel('Unfairness ↓', fontdict={'size':20, 'weight':'bold'})
    axs[i].set_title(dataset, fontdict={'size':25, 'weight':'bold'})
    axs[i].tick_params(labelsize=14)
    # Collect legend info from the first subplot that has one
    if handles is None:
        h, l = axs[i].get_legend_handles_labels()
        if h and l:
            handles, labels = h, l
    if handles is not None:
        fig.legend(handles, labels, 
                   loc='lower center', 
                   ncol=len(labels), 
                   bbox_to_anchor=(0.5, -0.095),
                   fontsize=18)

plt.tight_layout()
plt.savefig(f'{images_dir}/scatter_performance_vs_fairness.pdf', format='pdf', bbox_inches="tight")
# %%
def show_result(series, show_type='mean'):
    if show_type == 'mean':
        return f"{series.mean():.2f}±{series.std():.2f}"
    else:
        # median
        return f"{series.median():.2f}"
    
def select_best(series, selection_type):
    if selection_type == 'max':
        return series.max()
    elif selection_type == 'min':
        return series.min()
    else:
        # median
        return series.median()

# %%
def calculate_best_table(df, metrics, selected_metric, select_type='max'):
    rows = []
    for framework in df.framework.unique():
        rows.append(df.loc[(df.framework == framework) & (df.loc[df.framework == framework,selected_metric] == select_best(df.loc[df.framework == framework, selected_metric], select_type))])
    df = pd.concat(rows)
    return df[['framework'] + metrics]

def calculate_best_overall(df_by_dataset, metrics, selected_metric, select_type='max'):
    rows = []
    for dataset, df in df_by_dataset.items():
        df = calculate_best_table(df, metrics=metrics, selected_metric=selected_metric, select_type=select_type)
        df.insert(0, 'dataset', dataset)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
        
# %%
def calculate_summary_table_stat(df, metrics, latex_significance=False, significance_level=0.05, baseline_name='baseline', show_difference=False, bonferroni_correction=True, statistical_test=True, select_type='mean'):
    """
    calculates the mean +- std for all the frameworks and metrics given. it can also export the results to latex
    
    params:
        params:
        df : pd.DataFrame
            dataframe to use to make the statistical analysis. it should contain a column with "framework", where it contains the values for all the frameworks
        metrics : list
            metric to analyse
        latex_significance : boolean
            use latex significance style for papers
        significance_level : float
            significance level to consider
        bonferroni_correction : boolean
            Indicate whether or not to use Bonferroni Correction
    """
    summary_rows = []
    baseline = df.loc[df.framework == baseline_name]
    frameworks = [framework for framework in df.framework.unique() if framework != baseline_name]
    if bonferroni_correction:
        alpha = significance_level / (len(frameworks) * len(metrics))
    else:
        alpha = significance_level
    # baseline row
    baseline_row = {'framework': baseline_name}
    baseline_mean_results = {}
    for metric in metrics:
        if metric not in df.columns:
            continue
        base_mean = baseline[metric].mean()
        base_std = baseline[metric].std()
        import numpy as np
        baseline_mean_results[metric] = base_mean # save for the calculation
        baseline_row[metric] = show_result(baseline[metric], select_type)
    summary_rows.append(baseline_row)

    # frameworks rows
    for framework in frameworks:
        row = {'framework' : framework}
        framework_data = df.loc[df.framework == framework]
        differences = {'framework': '\scriptsize \% Change'}
        for metric in metrics:
            if metric not in df.columns:
                continue
            fw_mean = framework_data[metric].mean()
            fw_std = framework_data[metric].std()

            base_vals = baseline[metric]
            fw_vals = framework_data[metric]
            
            if show_difference:
                base_mean = baseline_mean_results[metric]
                percentage_diff = round((abs(fw_mean - base_mean) / base_mean) * 100, 2)
                signal = '+' if fw_mean >= base_mean else '-'
                if latex_significance:
                    percentage_diff_str = f'\scriptsize {signal}{abs(percentage_diff)}\\%'  # escape % for LaTeX
                else:
                    percentage_diff_str = f'{signal}{abs(percentage_diff)}'  # escape % for LaTeX
                differences[metric] = percentage_diff_str

            # Base LaTeX content (without significance)
            # if percentage_diff_str:
            #     val = (
            #         r'\makecell{$'
            #         f'{fw_mean:.3f}\\pm{fw_std:.3f}'
            #         r'$ \\ {\scriptsize['
            #         f'{percentage_diff_str}'
            #         r']}}'
            #     )
            # else:
            val = show_result(framework_data[metric], select_type)

            # make statistical test and apply significance formatting
            if statistical_test:
                print('>> stat test')
                stat, pval = statistical_test_repeated(base_vals, fw_vals, significance_level=significance_level)
                print('<< stat test')
                if pval < alpha:
                    if latex_significance:
                        row[metric] = '\\textbf{' + val + '}'
                    else:
                        row[metric] = f'{val}*'
                else:
                    row[metric] = val
            else:
                row[metric] = val
        
        summary_rows.append(row)
        if show_difference:
            summary_rows.append(differences)

    summary_df = pd.DataFrame(summary_rows)
    return summary_df

def calculate_summary_table_stat_2_setups(df, metrics, latex_significance=False, significance_level=0.05, baseline_name='baseline'):
    """
    calculates the mean +- std for the baseline, and the other framework,  and metrics given. 
    it can also export the results to latex
    the table contains the format
    | metric | baseline | other framework | p-value | significance | 
    
    params:
        df : pd.DataFrame
            dataframe to use to make the statistical analysis. it should contain a column with "framework", where it contains the values for all the frameworks
        metrics : list
            metric to analyse
        latex_significance : boolean
            use latex significance style for papers
        significance_level : float
            significance level to consider
    """
    summary_rows = []
    other_framework = list(set(df.framework.unique())-{baseline_name})[0]
    for metric in metrics:
        baseline_vals = df.loc[df.framework == baseline_name, metric]
        other_vals = df.loc[df.framework == other_framework, metric]

        baseline_mean, baseline_std = baseline_vals.mean(), baseline_vals.std()
        other_mean, other_std = other_vals.mean(), other_vals.std()

        stat, pval = stats.ttest_ind(baseline_vals, other_vals, equal_var=False)

        significance = '*' if pval < 0.05 else ''
        summary_rows.append({
            'metric' : metric,
            baseline_name : f'{baseline_mean:.3f} ± {baseline_std:.3f}',
            other_framework : f'{other_mean:.3f} ± {other_std:.3f}',
            'p-value' : f'{pval:.4f}',
            'significant': significance,
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df

# %%
def calculate_overall_table(df_by_dataset, datasets, metrics, significance_level=0.05, latex_significance=False, baseline_name='baseline', show_difference=False, export_path='', bonferroni_correction=True, select_type='mean', seeds_to_compare=None, statistical_test=True):
    results = []
    for dataset in datasets:
        try:
            print('> dataset', dataset)
            df = df_by_dataset[dataset]
            # first calculate the avg of the 5-folds of each experiment
            # if select_type == 'best':
            #     df = df.loc[
            #         df.groupby(['framework', 'experiment'])['cf fairaware-80-20'].idxmin()
            #     ].reset_index(drop=True)
            # else:
            #     df = df.groupby(by=['framework', 'experiment']).mean(numeric_only=True).reset_index()
            # # sort the values by seed
            # df = df.sort_values(by='seed')
            # if seeds_to_compare:
            #     df = df.loc[df.seed.isin(seeds_to_compare)]
            #     df = df.groupby(['framework', 'seed']).mean(numeric_only=True).reset_index()
            # else:
            #     # seeds_count = df.seed.value_counts()
            #     # print(seeds_count)
            #     df = df.groupby(['framework', 'seed']).mean(numeric_only=True).reset_index()
            #     # seeds = seeds_count.index[seeds_count == seed_per_exp]
            #     # df = df.loc[df.seed.isin(seeds)]
            summary = calculate_summary_table_stat(df, 
                metrics=metrics, 
                latex_significance=latex_significance, 
                significance_level=significance_level,
                baseline_name=baseline_name,
                show_difference=show_difference,
                bonferroni_correction=bonferroni_correction,
                statistical_test=statistical_test,
                select_type=select_type
            )
            summary.insert(0, 'dataset', dataset)
            results.append(summary)
        except Exception as e:
            print(e)
            continue
    overall_summary = pd.concat(results)
    if export_path:
        overall_summary.to_csv(f'{export_path}.csv', index=False)
    return overall_summary

# %%
fairness_metrics = ['demographic_parity', 'equalized_odds', 'abroca']
fairness_per_attribute = []
for fm in fairness_metrics:
    for attribute in ['gender', 'age', 'education', 'marital']:
        fairness_per_attribute.append(f'{attribute}_{fm}')
performance_metrics = ['f1', 'mcc', 'recall', 'num_evaluated_individuals', 'num_iterations']
data_metrics = ['data_%', 'sample_%', 'features_%']

def format_column (col):
    features_map = {
        'level_0' : 'Dataset',
        'framework' : 'Framework',
        'demographic_parity' : 'DP',
        'equalized_odds' : 'EO',
        'equal_opportunity' : 'EOp',
        'abroca' : 'ABROCA',
        'accuracy' : 'ACC',
        'balanced acc' : 'B. ACC',
        'roc_auc' : 'AUCROC',
        'positive_class_prop' : 'Positive Prop.',
        'sample_%' : 'Instances \%',
        'recall' : 'TPR'
    }
    if col in features_map:
        return '\\textbf{' + features_map[col] + '}'
    return '\\textbf{' + col.replace('_', ' ').replace('%', '\\%').upper() + '}'

# %%
show_result_type = 'median'
overall_summary = calculate_overall_table(
    df_by_dataset=df_by_dataset,
    datasets=datasets,
    metrics= performance_metrics + data_metrics,
    significance_level=0.01,
    latex_significance=False,
    baseline_name='Baseline',
    show_difference=False,
    export_path=None,
    bonferroni_correction=True,
    select_type=show_result_type, # ! remove to use all seeds available
    statistical_test=True
)

overall_summary.to_csv(f'current-stats-{show_result_type}.csv', index=False)

# %%
calculate_best_overall(df_by_dataset, performance_metrics + data_metrics, 'f1', 'max').to_csv('best_by_f1.csv', index=False)
calculate_best_overall(df_by_dataset, performance_metrics + data_metrics, 'num_evaluated_individuals', 'max').to_csv('best_by_num_evaluated_individuals.csv', index=False)

# %%
overall_summary = calculate_overall_table(
    df_by_dataset=df_by_dataset,
    datasets=datasets,
    metrics=fairness_metrics + performance_metrics + data_metrics,
    significance_level=0.05,
    latex_significance=True,
    baseline_name='Baseline',
    show_difference=True,
    export_path=f'../data/results-fairness/{experimentation_name}_geral_metrics',
    seed_per_exp=len(frameworks) if experimentation_name == 'cross-validation' else len(frameworks)*5,
    select_type='best'
)
overall_summary.columns = [format_column(col) for col in overall_summary.columns]
overall_summary = overall_summary.T
# %%
print(overall_summary.to_latex(
    bold_rows=False,
    index=True,
    column_format="|".join(["c"] * len(overall_summary.columns))
    ))


# %%
calculate_overall_table(
    df_by_dataset=df_by_dataset,
    datasets=datasets,
    metrics=fairness_per_attribute,
    significance_level=0.05,
    latex_significance=False,
    baseline_name='Baseline',
    show_difference=True,
    export_path=f'../data/results-fairness/{experimentation_name}_fairness',
    seed_per_exp=len(frameworks) if experimentation_name == 'cross-validation' else len(frameworks)*5
)


# %%
# print(overall_summary.to_latex(
#     bold_rows=True,
#     index=False))

# %%
overall_summary = calculate_overall_table(
    df_by_dataset=df_by_dataset,
    datasets=datasets,
    metrics=fairness_metrics + performance_metrics + data_metrics + ['data_%'],
    significance_level=0.05,
    latex_significance=False,
    baseline_name='Baseline',
    show_difference=True,
    export_path=f'../data/results-fairness/{experimentation_name}_geral_metrics',
    seed_per_exp=len(frameworks) if experimentation_name == 'cross-validation' else len(frameworks)*5
)

# %%
import numpy as np

# %%%

df_total = pd.concat(df_by_dataset.values(), ignore_index=True)
df_total['fairness'] = df_total[['demographic_parity', 'equalized_odds', 'abroca']].mean(axis=1)
print('Data vs Fairness Correlation')
for metric in ['fairness', 'demographic_parity', 'equalized_odds', 'abroca']:
    print(metric)
    print('Geral', np.corrcoef(df_total[metric], df_total['data_%'])[0][1])
    for framework in df_total.framework.unique():
        frame_df = df_total.loc[df_total.framework == framework]
        print('. ', framework, np.corrcoef(frame_df[metric], frame_df['data_%'])[0][1])