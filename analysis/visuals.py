"""
Script for plotting the graphs
"""

import pandas as pd
from utils import get_estimation_data, calculate_average_time
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
import json
from matplotlib.ticker import FuncFormatter

sns.set_style('whitegrid') 
sns.set_context('paper', font_scale=1.2)
plt.rcParams.update({
    # ---- Figure ----
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 500,          # Set default DPI for saving figures
    'savefig.format': 'pdf',     # Always save as a vector format (PDF or SVG)
    'figure.autolayout': True,

    # Axes spines
    'axes.linewidth': 2,      # thickness of all spines
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': 'black', # default spine color

    # Ticks
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'xtick.minor.width': 1,
    'ytick.minor.width': 1,

    # Grid
    'axes.grid': True,
    'grid.color': 'lightgrey',
    # 'grid.alpha': 0.5,
    'grid.linestyle': '-',   # major grid default
})


def plot_estimation_results(filepath, save_dir=None, performance_threshold=0.25, estimators_list=None):
    """ Plots the training curve of the estimators. 
    Args:
        filepath (str): Path to the CSV file containing the results (e.g. ".../estimators_info.csv").
        save_dir (str): Directory to save the plots.
        performance_threshold (float): Performance threshold for the estimators to be considered good.
    """
    df = pd.read_csv(filepath, index_col=0).T
    df.index.name = 'estimator'
    df = df.reset_index()
    for i, row in df.iterrows():
        estimator = row['estimator']
        if estimators_list is not None and estimator not in estimators_list:
            continue
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(15, 15))
        fig.suptitle(f'Estimator {estimator}', fontsize=25, fontweight='bold')

        training_individuals = pd.Series(ast.literal_eval(row['num_individuals_training']))
        estimated_individuals = pd.Series(ast.literal_eval(row['num_individuals_estimated']))
        train_generations = 10#len(training_individuals) - len(estimated_individuals) + 1
        lim_x = (-1, len(training_individuals)+1)

        metric = pd.Series(ast.literal_eval(row['prediction_error']))
        metric.index = np.arange(train_generations, train_generations + len(metric))
        estimated_values = metric.index[metric.values != -1]
        training_points = metric.index[metric.values == -1]
        percentage_retraining = pd.Series(ast.literal_eval(row['percentage_retrain']))
        percentage_retraining.index = np.arange(train_generations, len(percentage_retraining)+train_generations)
        estimated_individuals.index = np.arange(train_generations, len(estimated_individuals)+train_generations)

        axs[0].scatter(estimated_values, metric[estimated_values], label='Estimated')
        axs[0].scatter(training_points, metric[training_points], label='Training')
        axs[0].axhline(y=performance_threshold, color='black', linestyle='--', label='Performance Threshold')
        axs[0].axvline(x=train_generations, color='r', linestyle='--', label='Start of Estimation')
        axs[0].set_xlim(lim_x)
        axs[0].set_xlabel('Generation')
        axs[0].set_ylabel('Prediction Error')
        axs[0].set_title(f'Prediction Error')
        axs[0].legend()
        print('generations', len(ast.literal_eval(row['generation'])))
        print('train generations', train_generations)
        print('estimated generations', len(estimated_individuals))
        print('training gens', len(training_individuals))

        axs[1].scatter(estimated_individuals.index, estimated_individuals, label='No. Estimated')
        axs[1].scatter(training_individuals.index, training_individuals, label='No. Training')
        axs[1].set_title(f' Training vs Estimated Individuals')
        axs[1].axvline(x=train_generations, color='r', linestyle='--', label='Start of Estimation')
        axs[1].set_xlim(lim_x)
        axs[1].set_xlabel('Generation')
        axs[1].set_ylabel('Number of Individuals')
        axs[1].set_title(f'Number of Individuals Estimated vs Evaluated')
        axs[1].legend()

        only_estimated = estimated_individuals.cumsum() - training_individuals[train_generations-1:].sum()

        axs[2].plot(estimated_individuals.index, estimated_individuals.cumsum(), label='Sum Estimated Total')
        axs[2].plot(training_individuals[train_generations:].index, training_individuals[train_generations:].cumsum(), label='Sum Training From Estimation Start')
        axs[2].plot(training_individuals.cumsum().index, training_individuals.cumsum(), label='Sum Training From Beginning')
        # axs[2].plot(only_estimated.index, only_estimated, label='Sum Only Estimated')
        axs[2].set_xlim(lim_x)
        axs[2].axvline(x=train_generations, color='r', linestyle='--', label='Start of Estimation')
        axs[2].set_title(f'Number of Individuals Used')
        axs[2].set_xlabel('Generation')
        axs[2].set_ylabel('Cumulative Sum')
        axs[2].legend()

        axs[3].plot(percentage_retraining.index, percentage_retraining, label='Percentage Retraining', marker='o')
        axs[3].axvline(x=train_generations, color='r', linestyle='--', label='Start of Estimation')
        axs[3].set_xlim(lim_x)
        axs[3].set_xlabel('Generation')
        axs[3].set_ylabel('Percentage Retraining')
        axs[3].set_title(f'Percentage Retraining')
        axs[3].legend()
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}_estimator_{estimator}.pdf', bbox_inches='tight', format='pdf')
            plt.savefig(f'{save_dir}_estimator_{estimator}.png', bbox_inches='tight', format='png')
        plt.show()

def underline_text(s):
    return "".join(ch + "\u0332" for ch in s)

def plot_bar_comparison(per_framework, title='', key='', highlight_y=None, ax=None, save_path=None, palette=None):
    """
    plots the comparison between the selected {key} across the different frameworks

    params:
        per_framework: dict
            key=framework name
            value = values for each type of {key}
        title : str
            title of the plot
        key : str
            category to show
        highlight_y: list of str
            list with the y ticks to highlight or none
        ax : matplotlib ax
            show where to plot the figure
        save_path : str
            indicates the path where to save the image. DO NOT USE AT THE SAME TIME AS "ax". 
            it will add the format to the image
    """
    df = pd.DataFrame(per_framework).T
    df = df.reset_index().melt(id_vars='index', var_name=key, value_name='Frequency (%)')
    df.rename(columns={'index': 'Framework'}, inplace=True)

    # Plot with seaborn
    if ax == None:
        fig, ax = plt.subplots(figsize=(12, 7))

    sns.barplot(
        data=df,
        y=key,   # horizontal grouping axis
        x='Frequency (%)',     # numerical values
        hue='Framework',   # color/group variable
        orient='h',
        ax=ax,
        palette=palette
    )

    # annotate the bars
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.1f', label_type='edge', padding=3)
    # bars = ax.patches
    # for bar in bars:
    #     width = bar.get_width()
    #     y = bar.get_y() + bar.get_height() / 2
    #     ax.text(width + 0.1, y, f'{width:.1f}', va='center')

    if highlight_y:
        ticks = ax.get_yticks()
        new_labels = []

        for label in ax.get_yticklabels():
            text = label.get_text()
            if text in highlight_y:
                text = text.upper()
            new_labels.append(text)

        ax.set_yticks(ticks)
        ax.set_yticklabels(new_labels)

        for label in ax.get_yticklabels():
            if label.get_text().lower() in [h.lower() for h in highlight_y]:
                label.set_fontweight('bold')
                label.set_color('red')

    
    plt.legend()
    ax.set_xlim([0, 1])
    ax.set_title(title, fontdict={'weight': 'bold', 'size': 16})
    if save_path:
        plt.tight_layout()
        plt.savefig(f'{save_path}.pdf', format='pdf')

def boxplot_ax(data, x_variable, y_variable, ax, title='', xlim=None):
    """
    plot the boxplot for a given variable

    params:
        data: pd.DataFrame
            dataframe with the results to plot
        x_variable : str
            numeric variable to use plot
        y_variable : str
            variable to group by the experiments
        ax : matplotlib ax
            where to plot the graph
        title : str
            title for the experiment

    """
    # calculates the median max value to highlight it
    median = data.groupby(y_variable).median(numeric_only=True)[x_variable].max()
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
    if xlim:
        ax.set_xlim(xlim)
    ax.axvline(x=median, color='red', linestyle='--', linewidth=2, label='Max Median')
    ax.set_title(title, fontdict={'weight':'bold'})
    ax.set_ylabel(y_variable.replace('_', ' '), fontdict={'weight':'bold'})
    ax.set_xlabel(x_variable.replace('_', ' '), fontdict={'weight':'bold'})

def plot_estimation_process(df, ax, title):
    """ plots the estimation process given the dataframe with the EPM info across the generations"""
    # for each generation, it shows a stacked bar with the number of individuals trained and estimated
    print('>>> estimation evolution')
    print(df.head())
    # df = df.loc[df.generation.duplicated() == False]
    only_trained = df.loc[df.num_individuals_estimated == 0]
    with_estimation = df.loc[df.num_individuals_estimated > 0]
    ax.bar(with_estimation.generation, with_estimation['num_individuals_training'], label='Trained', color='cornflowerblue')
    ax.bar(only_trained.generation, only_trained['num_individuals_training'], label='Only Trained', color='darkorange')
    ax.bar(df.generation, df['num_individuals_estimated'], bottom=df['num_individuals_training'], label='Estimated', color='forestgreen')
    ax.set_xlabel('Generations', fontdict={'size':13})
    ax.set_ylabel('# Individuals', fontdict={'size':13})
    # ax.set_xticks(labels=np.arange(0, max_gen+10, 250), ticks=np.arange(0, max_gen+10, 250), fontdict={'size':10})
    ax.legend(loc='upper left')
    ax.set_title(title, fontdict={'size':14, 'weight' : 'bold'})
    print('<<< estimation evolution')
# def plot_estimation_process(df, ax, title):
#     print('>>> estimation evolution')
#     print(df.head())

#     dfg = (
#         df
#         .groupby('generation', as_index=False)
#         .agg(
#             trained=('num_individuals_training', 'sum'),
#             estimated=('num_individuals_estimated', 'sum')
#         )
#         .sort_values('generation')
#     )

#     ax.bar(
#         dfg['generation'],
#         dfg['trained'],
#         label='Trained',
#         color='cornflowerblue'
#     )

#     ax.bar(
#         dfg['generation'],
#         dfg['estimated'],
#         bottom=dfg['trained'],
#         label='Estimated',
#         color='forestgreen'
#     )

#     ax.set_xlabel('Generations', fontdict={'size': 13})
#     ax.set_ylabel('# Individuals', fontdict={'size': 13})
#     ax.legend(loc='upper left')
#     ax.set_title(title, fontdict={'size': 14, 'weight': 'bold'})

#     print('<<< estimation evolution')



def plot_estimation_individuals_evolution(setup, path, save_path=None):
    """ plots the evolution of the number of individuals trained and estimated across generations for several frameworks.
        It shows the evolution for each EPM 
    """

    last_exp = sorted(os.listdir(path))[1]
    with open(f'{path}/{last_exp}/edca/edca_fold3/estimators_info.json') as file:
        info = json.load(file)
    data = {}

    for model, values in info.items():
        try:
            data[model], initial_trained = get_estimation_data(values.copy())
        except Exception as e:
            print(e)
            continue

    # plot estimation vs training evolution over generations
    fig, axs = plt.subplots(nrows=len(data), ncols=1, figsize=(12, 4*len(data)))
    fig.suptitle(setup.replace('-', ' ').upper(), fontweight='bold', y=1.0001)
    for iter, (model, results) in enumerate(data.items()):
        try:
            plot_estimation_process(results, axs[iter], model)
        except Exception as e:
            print(e)
    fig.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/{setup}_evolution.pdf', format='pdf')
        plt.savefig(f'{save_path}/{setup}_evolution.png', format='png')

def barh_plot(ax, data, xlabel='', ylabel='', title='', color='skyblue', show_percentage=False):
    if show_percentage:
        total = sum(data.values())
        values = [round(val/total, 2) for val in data.values()]
    else:
        values = data.values()
    bars = ax.barh(data.keys(), values, color='skyblue')
    ax.bar_label(bars, padding=3, fmt='%.2f')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    

def plot_time_distribution_epms(framework, path, save_path=None):
    try:
        avg_time_info, avg_time_info_models = calculate_average_time(path)
    except Exception as e:
        print(e)
    # bar plot of the overall time
    fig, ax = plt.subplots(nrows=4, figsize=(10, 4*6))
    fig.suptitle(framework.replace('-', ' ').upper(), fontsize=20, fontweight='bold', y=1.00001)
    barh_plot(ax[0], avg_time_info, xlabel='Time (secs)', title='Time distribution in Estimation (All)', show_percentage=True)
    # pie chart of the distribution
    overall_evaluation_time_metrics = ['initial_evaluation', 'initial_train', 'evaluating_individuals', 'updating_epm', 'estimating_overall']
    overall_info = {key: avg_time_info[key] for key in overall_evaluation_time_metrics}
    ax[1].pie(overall_info.values(), labels=overall_info.keys(), autopct='%1.1f%%', startangle=90)
    ax[1].set_title(f'Time Distribution in Estimation (Overall)')
    
    barh_plot(ax[2], overall_info, xlabel='Time (secs)', title='Time distribution in Estimation', show_percentage=True)
    # inside estimation overall time
    estimation_vars = ['process_estimation', 'estimating_individuals']
    estimation_info = {key: avg_time_info[key] for key in estimation_vars}
    ax[3].pie(estimation_info.values(), labels=estimation_info.keys(), autopct='%1.1f%%', startangle=90)
    ax[3].set_title(f'Time Distribution in Estimation Process')
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(f'{save_path}/time_distribution_{framework}.png', bbox_inches='tight')
        plt.savefig(f'{save_path}/time_distribution_{framework}.pdf', format='pdf', bbox_inches='tight')

def plot_individuals_estimation_distribution(setup, path, save_path=None):
    """ Plot the distribution of estimated individuals across generations """
    last_exp = sorted(os.listdir(path))[-1]
    with open(f'{path}/{last_exp}/edca/edca_fold3/estimators_info.json') as file:
        info = json.load(file)
    data = {}
    for model, values in info.items():
        try:
            data[model], initial_trained = get_estimation_data(values.copy())
        except Exception as e:
            print(e)
            continue
    fig, axs = plt.subplots(nrows=len(data) // 3, ncols=3, figsize=(12, 7))
    fig.suptitle(setup.replace('-', ' ').upper(), fontweight='bold', y=1.0001)
    for i, (model, results) in enumerate(data.items()):
        trained = results.loc[results.num_individuals_estimated != 0, 'num_individuals_training'].sum()
        only_trained = results.loc[results.num_individuals_estimated == 0, 'num_individuals_training'].sum()
        estimated = results.num_individuals_estimated.sum()
        total = trained + estimated + only_trained
        axs[i // 3, i % 3].pie([trained / total, estimated / total, only_trained / total], 
                               labels=['Trained', 'Estimated', 'Only Trained'], 
                               autopct='%1.1f%%', 
                               startangle=90, 
                               colors=['cornflowerblue', 'forestgreen', 'darkorange'])
        axs[i // 3, i % 3].set_title(model, fontdict={'size': 16, 'weight': 'bold'})
    fig.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/{setup}_pie_summary.pdf', format='pdf')
        plt.savefig(f'{save_path}/{setup}_pie_summary.png', format='png')

