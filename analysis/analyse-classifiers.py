# %%
import pandas as pd
import matplotlib.pyplot as plt
from utils import best_individuals_overall, get_classifier
from visuals import plot_bar_comparison
from parameters import datasets, frameworks, images_dir, experimentation_name, frameworks_palette
import seaborn as sns

# %%

def map_classifier(classifier):
    if classifier == 'XGBClassifier':
        model = 'XGB'
    elif classifier == 'RandomForestClassifier':
        model = 'RF'
    elif classifier == 'ExtraTreesClassifier':
        model = 'ExtraTree'
    elif classifier == 'LogisticRegression':
        model = 'LR'
    elif classifier == 'KNeighborsClassifier':
        model = 'KNN'
    elif classifier == 'LGBMClassifier':
        model = 'LGBM'
    else:
        model = 'None'
    return model
def calculate_classifiers_occurrence(individuals, normalize=False):
    classifiers = []
    for individual in individuals:
        classifiers.append(map_classifier(get_classifier(individual)))
    series = pd.Series(classifiers).value_counts()
    if normalize:
        series = series / len(individuals)
    return series.to_dict()

# %%
results = {}
for dataset in datasets:
    per_framework = {}
    for framework, framework_path in frameworks.items():
        best_individuals = best_individuals_overall(f'{framework_path}/{dataset}')
        occurrence = calculate_classifiers_occurrence(best_individuals, normalize=True)
        per_framework[framework] = occurrence
    
    results[dataset] = per_framework
    plot_bar_comparison(per_framework, title=f'{dataset} Classifiers Comparison', key='Classifiers')

# %%
# fig, axs = plt.subplots(ncols=len(datasets), figsize=(len(datasets)*10, 7))
# for i, dataset in enumerate(sorted(results.keys())):
#     plot_bar_comparison(results[dataset], title=dataset, key='Classifiers', ax=axs[i])
# fig.suptitle('Final Classification Models', fontweight='bold', fontsize=25)
# plt.tight_layout()
# plt.savefig(f'{images_dir}/{experimentation_name}_classification_models_distribution.pdf', format='pdf')

# %%
fig, axs = plt.subplots(ncols=len(datasets),figsize=(len(datasets)*7, 5))

handles, labels = None, None

for i, dataset in enumerate(sorted(results.keys())):
    plot_bar_comparison(
        results[dataset],
        title=dataset,
        ax=axs[i],
        palette=frameworks_palette
    )
    axs[i].legend_.remove()
    # Collect legend info from the first subplot that has one
    if handles is None:
        h, l = axs[i].get_legend_handles_labels()
        if h and l:
            handles, labels = h, l

# Adjust layout and add a single legend below all subplots
# plt.tight_layout(rect=[0, 0.05, 1, 1.1])  # Make room at the bottom
if handles is not None:
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.085))
plt.tight_layout()
plt.savefig(f'{images_dir}/{experimentation_name}_classification_models_distribution.pdf', format='pdf')