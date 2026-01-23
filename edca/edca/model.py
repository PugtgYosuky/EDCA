import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


from sklearn import set_config
set_config(transform_output='pandas')

# get values from config

def get_integer_value(config):
    """ generates a integer value based on the range """
    return np.random.randint(low=config['min_value'], high=config['max_value'])


def get_float_value(config):
    """ generates a float value based on the range"""
    return np.round(np.random.uniform(low=config['min_value'], high=config['max_value']), 3)


def get_category_value(config):
    """ selects a categorical value based on the options available """
    return np.random.choice(config['possible_values'])


def model_parameters_mutation(model_config, model_parameters, prob_mutation):
    """
    Mutates the hyperparameters of a gene

    Parameters:
    ----------
    model_config : dict
        Gene to mutate

    model_parameters: dict
        Search space of the gene to mutate

    prob_mutation : float
        Probability of mutation

    Returns:
    -------
    dict
        Gene with mutated hyperparameters
    """
    for parameter in model_parameters.keys():
        # iterates over the model hyperparameters and mutates them based on the
        # prob_mutation
        if np.random.random() < prob_mutation:
            type_parameter = model_parameters[parameter]['value_type']
            if type_parameter == 'integer':
                model_config[parameter] = get_integer_value(model_parameters[parameter])
            if type_parameter == 'float':
                model_config[parameter] = get_float_value(model_parameters[parameter])
            if type_parameter == 'category':
                model_config[parameter] = get_category_value(model_parameters[parameter])
    return model_config


def generate_model_code(config):
    """ creates a new model based on the options available"""
    models = list(config.keys())
    model_name = np.random.choice(models)
    model = {
        model_name: model_parameters_mutation({}, config[model_name], 1.0)}
    return model


def models_mutations(config):
    """ mutates models based on the config"""
    def model_mutation(model_config, prob_model_mutation, prob_mutation):
        # random choice to choose if we are going to change the hyperparameters
        # or the model
        if np.random.random() < prob_model_mutation:
            # change the model used
            return generate_model_code(config)
        else:
            # hyperparameters mutation
            model_name = list(model_config.keys())[0]
            return {
                model_name: model_parameters_mutation(model_config[model_name],config[model_name],prob_mutation)}

    return model_mutation


def instantiate_model(model_config, seed=42):
    """ Instantiates the classifier given it's name and settings"""
    model_name = list(model_config.keys())[0]
    settings = model_config[model_name]
    if model_name == 'LogisticRegression':
        model = LogisticRegression(**settings, random_state=seed)

    elif model_name == 'KNeighborsClassifier':
        model = KNeighborsClassifier(**settings)

    elif model_name == 'SVC':
        model = SVC(**settings, random_state=seed, probability=True)

    elif model_name == 'GaussianNB':
        model = GaussianNB(**settings)

    elif model_name == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(**settings, random_state=seed)

    elif model_name == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor(**settings, random_state=seed)

    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(**settings, n_jobs=1, random_state=seed)

    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(**settings, n_jobs=1, random_state=seed)

    elif model_name == 'AdaBoostClassifier':
        model = AdaBoostClassifier(**settings, random_state=seed)

    elif model_name == 'AdaBoostRegressor':
        model = AdaBoostRegressor(**settings, random_state=seed)

    elif model_name == 'XGBClassifier':
        model = XGBClassifier(**settings, verbosity=0, n_jobs=1, random_state=seed)

    elif model_name == 'XGBRegressor':
        model = XGBRegressor(**settings, verbosity=0, n_jobs=1, random_state=seed)

    elif model_name == 'LGBMClassifier':
        model = LGBMClassifier(**settings, verbosity=-1, n_jobs=1, random_state=seed)

    elif model_name == 'LGBMRegressor':
        model = LGBMRegressor(**settings, verbosity=-1, n_jobs=1, random_state=seed)

    elif model_name == 'ExtraTreesClassifier':
        model = ExtraTreesClassifier(**settings, n_jobs=1, random_state=seed)

    elif model_name == 'ExtraTreesRegressor':
        model = ExtraTreesRegressor(**settings, n_jobs=1, random_state=seed)

    elif model_name == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(**settings, random_state=seed)

    elif model_name == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor(**settings, random_state=seed)

    else: # model not found
        raise ValueError(f"Model {model_name} not found")
    
    return model


def instantiate_imputer(imputer_config, seed=42):
    imputer_name = list(imputer_config.keys())[0]
    settings = imputer_config[imputer_name]

    if imputer_name == "SimpleImputer":
        imputer = SimpleImputer(**settings)
    elif imputer_name == "KNNImputer":
        imputer = KNNImputer(**settings)
    return imputer


def instantiate_encoder(encoder_config, seed=42):
    """ Instantiates the encoder from the given config """

    encoder_name = list(encoder_config.keys())[0]
    settings = encoder_config[encoder_name]

    if encoder_name == 'OneHotEncoder':
        encoder = OneHotEncoder(
            **settings,
            handle_unknown="ignore",
            sparse_output=False)
    elif encoder_name == 'LabelEncoder':
        encoder = LabelEncoder3Args(**settings)
    return encoder


def instantiate_scaler(scaler_name, seed=42):
    """ Intantiates the scaler from the given config """
    if scaler_name == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_name == "RobustScaler":
        scaler = RobustScaler()
    return scaler


def scalers_mutation(scaler):
    """ Mutation operator for the scaler component """
    options = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']
    index = options.index(scaler)
    operation = np.random.choice([np.add, np.subtract])
    new_index = operation(index, 1) % len(options)
    return options[new_index]


def generate_scaler_code():
    """ Generates the gene for the sclaer component """
    options = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']
    return np.random.choice(options)


class LabelEncoder3Args(BaseEstimator):
    def __init__(self, **kwargs):
        self.encoders = {}
        self.params = kwargs  # store any passed config if you want

    def fit(self, X, y=None):
        for column in X.columns:
            le = LabelEncoder()
            le.fit(X[column])
            self.encoders[column] = le
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for column in X.columns:
            le = self.encoders[column]
            known_classes = set(le.classes_)
            X[column] = X[column].apply(lambda val: le.transform([val])[0] if val in known_classes else (len(known_classes)+1))
        return X

def round_up(n):
    return round(n/10) * 10


def encode_numeric(feature_values, encoding_values):
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

class NumericEncoder(BaseEstimator):
    """Class to transform a numeric feature into a categorical one"""
    def __init__(self, numeric_encodings):
        """
        Params:
            numeric_encodings : dict
            Indicates the numeric feature to encode and the encoding cut-points to divide into the ranges
        """
        self.numeric_encodings = numeric_encodings

    def fit(self, X, y=None):
        # nothing to learn here
        return self

    def transform(self, X):
        X = X.copy()
        # transforms the given features
        for item, values in self.numeric_encodings.items():
            if item in X.columns:
                X[item] = encode_numeric(X[item], values)
        return X


def create_preprocessing_pipeline(
        selected_features,
        pipeline_config,
        individual,
        numeric_encodings=None):
    """
    Create the preprocessing pipeline based on the individual and the data characteristics

    It adds different transformer to the pipeline, based on the data types presented and
    data characteristics such as null values.

    Parameters:
    ----------
    pipeline_config : dict
        Data characteristics

    individual : dict
        Configuration of the individual / pipeline

    Returns:
    -------
        numpy.array
            predictions
    """
    # get features from the pipeline config intersected with the selected features
    
    column_transformer_steps = []
    numerical_columns = list(set(pipeline_config['numerical_columns']).intersection(set(selected_features)))
    numerical_with_nans = list(set(pipeline_config['numerical_with_nans']).intersection(set(selected_features)))
    categorical_columns = list(set(pipeline_config['categorical_columns']).intersection(set(selected_features)))
    categorical_with_nans = list(set(pipeline_config['categorical_with_nans']).intersection(set(selected_features)))
    binary_columns = list(set(pipeline_config['binary_columns']).intersection(set(selected_features)))
    binary_with_nans = list(set(pipeline_config['binary_with_nans']).intersection(set(selected_features)))

    if len(numerical_columns) > 0:
        # numerical transformer if it has numerical fuatures
        num_steps = []

        if len(numerical_with_nans) > 0:
            # if it has numerical features with nans
            numerical_imputer_config = individual['numerical-imputer']
            num_imputer = instantiate_imputer(numerical_imputer_config)
            num_steps.append(('num_imputer', num_imputer))

        scaler_config = individual['scaler']
        scaler = instantiate_scaler(scaler_config)
        num_steps.append(('scaler', scaler))
        numerical_pipeline = Pipeline(steps=num_steps)
        column_transformer_steps.append(
            ('num', numerical_pipeline, numerical_columns))

    if len(categorical_columns) > 0:
        # categorical transformer if it has categorical features
        cat_steps = []

        if len(categorical_with_nans) > 0:
            # imputer values if it has nans
            cat_imputer_config = individual['categorical-imputer']
            cat_imputer = instantiate_imputer(cat_imputer_config)
            cat_steps.append(('cat_imputer', cat_imputer))

        encoder_config = individual['encoder']
        encoder = instantiate_encoder(encoder_config)
        cat_steps.append(('encoder', encoder))
        cat_pipeline = Pipeline(steps=cat_steps)
        column_transformer_steps.append(
            ('cat', cat_pipeline, categorical_columns))

    if len(binary_columns) > 0:
        # binary transformer if it has binary features
        bin_steps = []
        if len(binary_with_nans) > 0:
            # imputer values
            cat_imputer_config = individual['categorical-imputer']
            cat_imputer = instantiate_imputer(cat_imputer_config)
            bin_steps.append(('bin_imputer', cat_imputer))
        # hard coded
        bin_steps.append(('bin_encoder', LabelEncoder3Args()))
        bin_pipeline = Pipeline(steps=bin_steps)
        column_transformer_steps.append(('bin', bin_pipeline, binary_columns))

    # merge all the transformers
    pipeline = ColumnTransformer(
        transformers=column_transformer_steps,
        verbose_feature_names_out=False,
        remainder='drop',
    )
    # add numeric encoder
    if numeric_encodings:
        pipeline = Pipeline(
            steps=[
                ('numeric_encoder', NumericEncoder(numeric_encodings=numeric_encodings)),
                ('optimised_pipeline', pipeline)
            ]
        )
    return pipeline
