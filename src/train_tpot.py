import os
import pandas as pd
import json
import numpy as np
import random as rnd 
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

import warnings
warnings.filterwarnings("ignore")

from utils import append_metrics

from edca.encoder import NpEncoder

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tpot.export_utils import set_param_recursive

def instantiate_tpot_model(pipeline_path, seed=42):
    with open(pipeline_path, 'r') as f:
        content = f.read()
    # get the code for the pipeline
    pipeline = content.split('exported_pipeline')[1].split('\n#')[0]
    pipeline = 'exported_pipeline' + pipeline
    # execute the pipeline
    exec(pipeline, globals())
    # fix seed
    if hasattr(exported_pipeline, 'steps'):
        set_param_recursive(exported_pipeline.steps, 'random_state', seed)
    elif hasattr(exported_pipeline, 'steps'): 
        setattr(exported_pipeline, 'random_state', seed)
    
    return exported_pipeline


def train_tpot(name, pipeline, X_train, y_train, X_test, y_test, results, seed, original_size):
    # encode y
    y_encoder = LabelEncoder()
    train_y = y_encoder.fit_transform(y_train)

    # train pipeline
    pipeline.fit(X_train, train_y)
    # predict
    preds = y_encoder.inverse_transform(pipeline.predict(X_test))
    preds_proba = pipeline.predict_proba(X_test)

    append_metrics(
        automl_framework=name,
        results=results,
        y_test=y_test,
        preds=preds,
        proba_preds=preds_proba,
        final_data_size=X_train.shape,
        original_data_size=original_size,
    )


def evaluate_with_data(X_train, y_train, X_test, y_test, config, results, seed, edca_results, fold, pipeline_path):
    X_train_edca, X_val_edca, y_train_edca, y_val_edca = train_test_split(
        X_train, y_train,
        shuffle=True,
        stratify=y_train,
        test_size=config['validation_size'],
        random_state=seed,
    )
    x_train = X_train_edca.copy()
    y_train = y_train_edca.copy()
    x_test = X_test.copy()
    try:
        if 'sample' in edca_results['evo_best'][fold] and 'features' not in edca_results['evo_best'][fold]:
            # only sampling
            x_train = X_train_edca.iloc[edca_results['evo_best'][fold]['sample']]
            y_train = y_train_edca.iloc[edca_results['evo_best'][fold]['sample']]
            
        if 'features' in edca_results['evo_best'][fold] and 'sample' not in edca_results['evo_best'][fold]:
            # only FS
            features = X_train_edca.columns[edca_results['evo_best'][fold]['features']]
            x_train = X_train_edca[features]
            y_train = y_train_edca
            x_test = X_test[features]

        if 'sample' in edca_results['evo_best'][fold] and 'features' in edca_results['evo_best'][fold]:
            # IS and FS
            features = X_train_edca.columns[edca_results['evo_best'][fold]['features']]
            x_train = X_train_edca.iloc[edca_results['evo_best'][fold]['sample']][features]
            y_train = y_train_edca.iloc[edca_results['evo_best'][fold]['sample']]
            x_test = X_test[features]
            
    
        train_tpot(
            pipeline=instantiate_tpot_model(pipeline_path, seed),
            name='tpot_with_edca',
            X_train=x_train, 
            y_train=y_train, 
            X_test=x_test, 
            y_test=y_test, 
            results=results, 
            seed=seed,
            original_size=X_train.shape)
    except Exception as e:
        # print(e)
        append_metrics(
            automl_framework='tpot_with_edca',
            results=results,
            y_test=y_test,
            preds=None,
            proba_preds=None,
            final_data_size=None,
            original_data_size=None,
            error=True)

def main(config, edca_results, save_path, pipeline_path):
    seed = config['seed']
    rnd.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(config['dataset'])
    X = df.copy()
    y = X.pop('class')

    results = {
        'dataset' : config['dataset'],
    }

    # stratified K fold
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_indexes, test_indexes) in enumerate(kfold.split(X, y)):
        print('FOLD:', fold+1)
        # get train and test data
        X_train = X.iloc[train_indexes]
        y_train = y.iloc[train_indexes]
        X_test = X.iloc[test_indexes]
        y_test = y.iloc[test_indexes]
        aux_path = os.path.join(pipeline_path, f'best_pipeline_fold{fold+1}.py')
        evaluate_with_data(X_train, y_train, X_test, y_test, config, results, seed, edca_results, fold, aux_path)

    with open(save_path, 'w') as file:
        json.dump(results, file, indent=3, cls=NpEncoder)

if __name__ == '__main__':

    datasets = ['Amazon_employee_access']#['adult', 'Australian', 'cnae-9', 'credit-g', 'kr-vs-kp', 'mfeat-factors']
    edca_framework = 'edca-1-0-0'
    save_path = os.path.join('..', 'thesis-results', f'tpot-{edca_framework}-results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for dataset in datasets:
        for run in range(30):
            print(dataset, 'RUN:', run)
            source_path = os.path.join('..', 'thesis-results', 'datasets-divided')
            with open(os.path.join(source_path, edca_framework, dataset, f'run_{run}', 'config.json')) as file:
                config = json.load(file)

            with open(os.path.join(source_path, edca_framework, dataset,  f'run_{run}', 'results.json')) as file:
                edca_results = json.load(file)

            tpot_pipeline_path = os.path.join(source_path, 'tpot', dataset, f'run_{run}', 'tpot')
            save_file = os.path.join(save_path, f'results_{dataset}_{run}.json')

            main(config, edca_results, save_file, tpot_pipeline_path)
