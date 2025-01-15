import os
import pandas as pd
import json
from flaml.automl.data import DataTransformer
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random as rnd 
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils import append_metrics

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from edca.encoder import NpEncoder

def instantiate_flaml_model(learner, config, seed=42):
    if learner == 'lgbm':
        model = LGBMClassifier(**config, random_state=seed)
    elif learner == 'xgboost' or learner == 'xgb_limitdepth':
        model = XGBClassifier(**config, random_state=seed)
    elif learner == 'rf':
        if 'max_leaves' in config:
            config['max_leaf_nodes'] = config.pop('max_leaves')
        model = RandomForestClassifier(**config, random_state=seed)
    elif learner == 'lrl1' or learner == 'lrl2':
        model = LogisticRegression(**config, random_state=seed)
    elif learner == 'kneighbor':
        model = KNeighborsClassifier(**config)
    elif learner == 'extra_tree':
        if 'max_leaves' in config:
            config['max_leaf_nodes'] = config.pop('max_leaves')
        model = ExtraTreesClassifier(**config, random_state=seed)
    else:
        model = None
    return model

def train_flaml(name, model, X_train, y_train, X_test, y_test, results, seed, original_size):
    # encode y
    y_encoder = LabelEncoder()
    train_y = y_encoder.fit_transform(y_train)

    # transform data
    flaml_transformer = DataTransformer()
    X_transformed, y_transformed = flaml_transformer.fit_transform(X_train, train_y, task='classification')
    X_test_transformed = flaml_transformer.transform(X_test)
    model.fit(X_transformed, y_transformed)

    preds = y_encoder.inverse_transform(model.predict(X_test_transformed))
    preds_proba = model.predict_proba(X_test_transformed)

    append_metrics(
        automl_framework=name,
        results=results,
        y_test=y_test,
        preds=preds,
        proba_preds=preds_proba,
        final_data_size=X_train.shape,
        original_data_size=original_size,
    )


def evaluate_with_data(X_train, y_train, X_test, y_test, config, results, seed, edca_results, flaml_results, fold):
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
            
    
        train_flaml(
            name='flaml_with_edca',
            model=instantiate_flaml_model(
                learner=flaml_results['flaml_best_learner'][fold], 
                config=flaml_results['flaml_best_config'][fold],
                seed=seed),
            X_train=x_train, 
            y_train=y_train, 
            X_test=x_test, 
            y_test=y_test, 
            results=results, 
            seed=seed,
            original_size=X_train.shape)
    except:
        append_metrics(
            automl_framework='flaml_with_edca',
            results=results,
            y_test=y_test,
            preds=None,
            proba_preds=None,
            final_data_size=None,
            original_data_size=None,
            error=True)

def main(config, edca_results, flaml_results, save_path):
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

        evaluate_with_data(X_train, y_train, X_test, y_test, config, results, seed, edca_results, flaml_results, fold)

    with open(save_path, 'w') as file:
        json.dump(results, file, indent=3, cls=NpEncoder)

if __name__ == '__main__':

    datasets = ['bank-marketing', 'Amazon_employee_access']#['adult', 'Australian', 'cnae-9', 'credit-g', 'kr-vs-kp', 'mfeat-factors']
    edca_framework = 'edca-1-0-0'
    save_path = os.path.join('..', 'thesis-results', f'flaml-{edca_framework}-results')
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

            with open(os.path.join(source_path, 'flaml', dataset, f'run_{run}', 'results.json')) as file:
                flaml_results = json.load(file)
            save_file = os.path.join(save_path, f'results_{dataset}_{run}.json')

            main(config, edca_results, flaml_results, save_file)