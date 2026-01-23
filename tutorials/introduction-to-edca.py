"""
🧬 Getting Started with EDCA

Welcome! In this tutorial, we will learn how to use EDCA, an AutoML framework that focuses on your data's unique characteristics to build better models.

Instead of just searching for a model, EDCA treats data as a first-class citizen, automatically evolving end-to-end pipelines that clean, reduce, and optimize themselves specifically for your dataset.

Run this script with: "python introduction-to-edca.py" or use the docker provided

"""

# * imports * 

# to import edca from the original src, without installing it via pip
# import sys
# sys.path.append('../edca')
import numpy as np
import random
import os
import pandas as pd
import datetime
from edca.evodata import DataCentricAutoML
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split # to evaluate the framework
from sklearn import metrics # for calculating metrics

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == '__main__':

    seed = 42 # ! change accordingly
    # fix seed for reproducibility
    fix_seed(seed=seed)

    # * load dataset *
    # Start by getting the data to feed EDCA. EDCA receives dataset in the pandas DataFrame format. 
    # You can fetch a dataset from OpenML to continue (example 1) or continue with your one dataset (example 2). 
    # You need to divide the data into the dataframe (X) and the target series (y).

    # # example 1 - fetch a classification dataset from OpenML
    # data_id = 31 # this example used the credit-g dataset (https://www.openml.org/search?type=data&sort=runs&status=active&id=31) 
    # X, y = fetch_openml(data_id=151, return_X_y=True, as_frame=True)

    # example 2 - your own dataset
    # ! change data path accordingly
    data_path = os.path.join('data/datasets/Australian.csv') # path to your dataset
    X = pd.read_csv(data_path) # load dataframe
    y = X.pop('class') # divide it into data and target

    # * split dataset into train and test for later evaluation of best solution*
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # * initialize EDCA *
    # in this example, most EDCA parameter are on their default settings. See all available parameter in EDCA/edca/README.md

    # create a folder to store all information regarding the optimization
    save_path = 'logs' # ! change accordingly
    os.makedirs(save_path, exist_ok=True)

    # initialize the class
    automl = DataCentricAutoML(
        task='classification', # detail the ML task
        seed=seed, # ensure reproducibility
        metric='f1', # specify the search metric
        time_budget=-1, # specify the time budget in seconds, -1 indicates no time limit, using the iterations as stop criteria
        n_iterations=50, # specify the number of iterations
        log_folder_name=f'{save_path}/experiment-{datetime.datetime.now()}', # specify the log folder to store information
        use_sampling=True, # use sampling to speed up the search
        use_feature_selection=True # use feature selection to speed up the search and improve the model generalization
    )

    # * optimize the ML pipeline with EDCA*
    automl.fit(X_train, y_train)

    # * analyse the best solution *
    print('Best solution config:', automl.best_individual)
    print('Pipeline config:', automl.pipeline_estimator)
    print('Data processing pipeline:', automl.pipeline_estimator.pipeline)
    print('Classification Model:', automl.pipeline_estimator.model)
    print('Best solution results on optimization:', automl.search_algo.best_fitness_params)

    # * analyze selected data *
    final_X, final_y = automl.get_final_data()
    print('Original Train dataset:', X_train.shape)
    print('EDCA internal train dataset', automl.internal_x_train.shape)
    print('EDCA selected dataset:', automl.get_final_data_shape())
    print('Selected Features:', final_X.columns)

    # * make prediction and assess prediction power *
    preds = automl.predict(X_test)
    preds_proba = automl.predict_proba(X_test)
    print(metrics.classification_report(y_test, preds))





