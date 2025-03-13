from sklearn.base import BaseEstimator
import numpy as np
from edca.model import create_preprocessing_pipeline, instantiate_model
from sklearn.preprocessing import LabelEncoder
from flaml import AutoML
import time
import json
import os
import time
# setup config
from sklearn import set_config
set_config(transform_output='pandas')

class PipelineEstimator(BaseEstimator):
    """
    Class to execute a given individual/ pipeline

    """

    def __init__(self, individual_config, pipeline_config=None, seed=None, individual_id=None):
        """
        Initialization

        Assigns the given parameters

        Parameters:
        ----------
        individual_config : dict
            Configuration of an individual

        pipeline_config : dict
            Characteristics of the data to create the adequate pipeline

        use_sampling : bool
            Tell either should be use sampling to train the individual/pipeline or not

        use_feature_selection : bool
            Tell either should be use feature selection to train the individual/pipeline or not

        Returns:
        -------
            None
        """
        self.individual_config = individual_config
        self.pipeline_config = pipeline_config
        self.seed = seed
        self.individual_id = individual_id
        # time variables
        self.model_training_time = None
        self.data_processing_time = None
        self.train_time = None # all the time spent in training the model (model training time + data processing time)
        self.prediction_time = None # time spent in predicting the model


    def fit(self, X, y, seed=42):
        """
        Train the individual / pipeline based on the received data

        Selects the data to use, in case of applying sampling. If the pipeline_config parameters is null,
        it analysis the dataset to see which pipeline is required. It instantiate the pipeline and trains it.

        Parameters:
        ----------
        X : pandas.DataFrame
            Data / features to train the pipeline

        y : pandas.Series
            Target column

        Returns:
        -------
            self
        """
        # analyse dataset if pipeline config is null
        if self.pipeline_config is None:
            self.pipeline_config = dataset_analysis(X)
            self.pipeline_config['seed'] = seed
        self.X = X.copy()
        self.y = y.copy()
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # start data processing
        start_time = time.time()
        self.X_train, self.y_train, self.selected_features = get_selected_data(self.X_train, self.y_train, self.individual_config)

        # create preprocessing pipeline
        self.pipeline = create_preprocessing_pipeline(
            selected_features=self.selected_features,
            pipeline_config=self.pipeline_config,
            individual=self.individual_config
        )

        self.X_training = self.pipeline.fit_transform(self.X_train).copy()

        # encode the target class for classification tasks
        if self.pipeline_config['task'] == 'classification':
            self.y_encoder = LabelEncoder()
            self.y_train_encoded = self.y_encoder.fit_transform(self.y_train)

        # end data processing
        self.data_processing_time = time.time() - start_time

        # create the model
        start_time = time.time()
        if self.pipeline_config.get('flaml_ms', False) == False:
            self.model = instantiate_model(self.individual_config.get('model'), seed=self.pipeline_config.get('seed'))
            self.model.fit(self.X_training, self.y_train_encoded)

        else:
            # use FLAML
            settings = {
                'task': self.pipeline_config['task'],
                'metric': self.pipeline_config['search_metric'],
                'keep_search_state': True,
                'sample': False,
                'retrain_full': False,
                'model_history': False,
                'early_stop': True,
                'n_jobs': 1,
                'skip_transform': True,
                'verbose': 0,
                'seed': self.pipeline_config['seed'],
                'max_iter' : 30, # !CHANGE THIS!
                'auto_augment' : False,
                'estimator_list' : ['lgbm', 'xgboost', 'xgb_limitdepth', 'rf', 'lrl1', 'lrl2', 'kneighbor', 'extra_tree'],
                'time_budget' : self.pipeline_config['time_budget'] - (time.time() - self.pipeline_config['start_datetime']),
                'log_type' : 'all',
                'log_file_name' : os.path.join(self.pipeline_config['flaml_save_dir'], f'individual_{self.individual_id}.log'),
            }

            if 'flaml_estimator' in self.individual_config:
                # retrain best FLAML model
                settings['estimator_list'] = [self.individual_config['flaml_estimator']]
                settings['starting_points'] = {self.individual_config['flaml_estimator'] : json.loads(self.individual_config['flaml_estimator_config'])}
                settings['max_iter'] = 1
                settings['log_file_name'] = ''
                
            self.model = AutoML(**settings)
            self.model.fit(X_train=self.X_training, y_train=self.y_train_encoded, seed=self.pipeline_config['seed'])
        
        self.model_training_time = time.time() - start_time
        self.train_time = self.model_training_time + self.data_processing_time
        return self

    def predict(self, X):
        """
        Predicts the output of a given data

        It preprocesses the given data and predicts the results of the model

        Parameters:
        ----------
        X : pandas.DataFrame
            Data to predict

        Returns:
        -------
            numpy.array
                predictions
        """
        # start prediction time
        start_time = time.time()
        X_test = X.copy()
        X_test = X_test[self.selected_features]
        X_test = self.pipeline.transform(X_test)
        preds = self.model.predict(X_test)
        # decode the target class for classification tasks
        if self.pipeline_config['task'] == 'classification':
            preds = self.y_encoder.inverse_transform(preds)
        self.prediction_time = time.time() - start_time
        return preds

    def predict_proba(self, X):
        """
        Predicts the probability of each output class of the given data

        It preprocesses the given data and predicts the results of the model

        Parameters:
        ----------
        X : pandas.DataFrame
            Data to predict

        Returns:
        -------
            numpy.array
                predictions
        """
        X_test = X.copy()
        X_test = X_test[self.selected_features]
        X_test = self.pipeline.transform(X_test)
        return self.model.predict_proba(X_test)

    def get_best_sample_data(self):
        if 'sample' in self.individual_config:
            # check if all values are 0 or 1 = binary representation
            if len(set(self.individual_config['sample'] + [0, 1])) == 2:
                indexes_vector = self.individual_config.get('sample')
                indexes_vector = np.array(indexes_vector).astype(bool)
                x = self.X.loc[indexes_vector].copy()
                y = self.y.loc[indexes_vector].copy()
            else:
                # integer representation
                x = self.X.iloc[self.individual_config['sample']].copy()
                y = self.y.iloc[self.individual_config['sample']].copy()
            return x, y
        else:
            return self.X, self.y


def get_selected_data(X_train, y_train, individual_config):
    # select samples to use in training
        if 'sample' in individual_config:
            # check if all values are 0 or 1 = binary representation
            if len(set(individual_config['sample'] + [0, 1])) == 2:
                indexes_vector = individual_config.get('sample')
                indexes_vector = np.array(indexes_vector).astype(bool)
                X_train = X_train.loc[indexes_vector].copy()
                y_train = y_train.loc[indexes_vector].copy()
            else:
                # integer representation
                X_train = X_train.iloc[individual_config['sample']].copy()
                y_train = y_train.iloc[individual_config['sample']].copy()

        # select features to use in training
        selected_features = list(X_train.columns)
        if 'features' in individual_config:
            selected_features = X_train.columns[individual_config['features']]
            X_train = X_train[selected_features].copy()
        return X_train, y_train, selected_features

def dataset_analysis(df):
    """
    Analysis the given data

    It analysis the data types of the given columns of the dataframe

    Parameters:
    ----------
    df : pandas.DataFrame
        Data to analyse

    Returns:
    -------
        dict
            contains lists of features containing the different data types present in the dataset
    """

    print('>>> Dataset Analysis')
    # select null cols
    null_cols = list(df.columns[df.isnull().sum() == len(df)])
    # select columns with nan that not belong to null_cols
    columns_with_nans = [col for col in list(
        df.columns[df.isnull().any()]) if col not in null_cols]
    # select numerical columns dat not belong to null cols
    numerical_columns = [col for col in df.select_dtypes(
        include=['number']).columns.tolist() if col not in null_cols]
    # select other columns, i.e, binary and categorical columns
    other_columns = [
        col for col in df.columns if col not in numerical_columns and col not in null_cols]
    categorical_columns = []
    id_columns = []
    binary_columns = []

    # check bools with empty columns
    for column in other_columns:
        aux = df[column].dropna()
        if aux.nunique() == 2:
            binary_columns.append(column)
        elif aux.nunique() == len(aux):
            id_columns.append(column)
        else:
            categorical_columns.append(column)

    # combines the informations of the nans and type of features to calculate
    # the features seperated by type with nans
    numerical_with_nans = list(set(numerical_columns) & set(columns_with_nans))
    binary_with_nans = list(set(binary_columns) & set(columns_with_nans))
    categorical_with_nans = list(
        set(categorical_columns) & set(columns_with_nans))
    # create the pipeline config with the types of features separated by
    # characteristics
    pipeline_config = {
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns,
        'id_columns': id_columns,
        'binary_columns': binary_columns,
        'numerical_with_nans': numerical_with_nans,
        'categorical_with_nans': categorical_with_nans,
        'binary_with_nans': binary_with_nans,
        'null_cols': null_cols
    }
    print('<<< Dataset Analysis')
    return pipeline_config
