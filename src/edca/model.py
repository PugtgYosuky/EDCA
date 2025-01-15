import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer


from sklearn import set_config
set_config(transform_output='pandas')

# get values from config

def get_integer_value(config):
    """ generates a integer value based on the range """
    return np.random.randint(low=config['min_value'], high=config['max_value'])


def get_float_value(config):
    """ generates a float value based on the range"""
    return np.random.uniform(low=config['min_value'], high=config['max_value'])


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
                model_config[parameter] = get_integer_value(
                    model_parameters[parameter])
            if type_parameter == 'float':
                model_config[parameter] = get_float_value(
                    model_parameters[parameter])
            if type_parameter == 'category':
                model_config[parameter] = get_category_value(
                    model_parameters[parameter])
    return model_config


def generate_model_code(config):
    """ creates a new model based on the options available"""
    models = list(config.keys())
    model_name = np.random.choice(models)
    model = {
        model_name: model_parameters_mutation(
            {}, config[model_name], 1.0)}
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
                model_name: model_parameters_mutation(
                    model_config[model_name],
                    config[model_name],
                    prob_mutation)}

    return model_mutation

def instantiate_data_augmentation(augmentation_config, metadata, seed=42):
    model_name = list(augmentation_config.keys())[0]
    settings = augmentation_config[model_name].copy()
    settings['enforce_rounding'] = bool(settings['enforce_rounding'])
    settings['enforce_min_max_values'] = bool(settings['enforce_min_max_values'])
    sample_percentage = settings.pop('sample_percentage')
    if model_name == "GaussianCopulaSynthesizer":
        model = GaussianCopulaSynthesizer(metadata=metadata, **settings)
    elif model_name == "CTGANSynthesizer":
        settings = settings.copy()
        settings['verbose'] = False
        model = CTGANSynthesizer(metadata=metadata, **settings)
    elif model_name == "TVAESynthesizer":
        model = TVAESynthesizer(metadata=metadata, **settings)
    return model, sample_percentage

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
        model = RandomForestClassifier(**settings, random_state=seed)

    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(**settings, random_state=seed)

    elif model_name == 'AdaBoostClassifier':
        model = AdaBoostClassifier(**settings, random_state=seed)

    elif model_name == 'AdaBoostRegressor':
        model = AdaBoostRegressor(**settings, random_state=seed)

    elif model_name == 'XGBClassifier':
        model = XGBClassifier(**settings, random_state=seed, verbosity=0)

    elif model_name == 'XGBRegressor':
        model = XGBRegressor(**settings, random_state=seed, verbosity=0)

    elif model_name == 'LGBMClassifier':
        model = LGBMClassifier(**settings, random_state=seed, verbosity=-1)

    elif model_name == 'LGBMRegressor':
        model = LGBMRegressor(**settings, random_state=seed, verbosity=-1)

    elif model_name == 'ExtraTreesClassifier':
        model = ExtraTreesClassifier(**settings, random_state=seed)

    elif model_name == 'ExtraTreesRegressor':
        model = ExtraTreesRegressor(**settings, random_state=seed)

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
    """ Intantiates the encoder from the given config """

    encoder_name = list(encoder_config.keys())[0]
    settings = encoder_config[encoder_name]

    if encoder_name == 'OneHotEncoder':
        encoder = OneHotEncoder(
            **settings,
            handle_unknown="ignore",
            sparse=False)
    elif encoder_name == 'LabelEncoder':
        encoder = LabelEncoder(**settings)
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
    def __init__(self):
        # super().__init__(**kwargs)
        self.encoders = {}

    def fit(self, X, y=None):
        # 1 encoder per binary column
        for column in X.columns:
            aux = {}
            encoder = LabelEncoder()
            encoder.fit(X[column])
            aux['encoder'] = encoder
            aux['classes_'] = encoder.classes_
            self.encoders[column] = aux
        return self

    def transform(self, X, y=None):
        # transforms the columns with the fitted information
        for column in X.columns:
            encoder = self.encoders[column]['encoder']
            X[column] = encoder.transform(X[column])
        return X


def create_preprocessing_pipeline(
        selected_features,
        pipeline_config,
        individual):
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
    # get features from the pipeline config interseted with the selected
    # features
    column_transformer_steps = []
    numerical_columns = list(
        set(pipeline_config['numerical_columns']).intersection(set(selected_features)))
    numerical_with_nans = list(
        set(pipeline_config['numerical_with_nans']).intersection(set(selected_features)))
    categorical_columns = list(
        set(pipeline_config['categorical_columns']).intersection(set(selected_features)))
    categorical_with_nans = list(set(
        pipeline_config['categorical_with_nans']).intersection(set(selected_features)))
    binary_columns = list(
        set(pipeline_config['binary_columns']).intersection(set(selected_features)))
    binary_with_nans = list(
        set(pipeline_config['binary_with_nans']).intersection(set(selected_features)))

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
    return pipeline
