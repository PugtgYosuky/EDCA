import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from edca.evodata import *
from edca.model import *
import os
import json
from edca.evolutionary_algorithm import EvolutionarySearch
from edca.encoder import NpEncoder
from edca.ea import mutation_individuals, mutation_sampling_component, sample_class_balance_mutation, uniform_crossover, points_crossover, generate_sampling_component
from edca.estimator import PipelineEstimator, dataset_analysis, get_selected_data
from datetime import datetime
from edca.utils import error_metric_function

class DataCentricAutoML(BaseEstimator):
    """ EDCA class """

    def __init__(self, task, metric='mcc', validation_size=0.25,
                 n_iterations=10,
                 time_budget=60,
                 binary_sampling_component=True,
                 automatic_data_optimization=True,
                 use_sampling=False,
                 use_feature_selection=False,
                 use_data_augmentation=False, 
                 prob_mutation=0.3,
                 prob_mutation_model=0.5,
                 prob_crossover=0.7,
                 tournament_size=3,
                 elitism_size=1,
                 population_size=10,
                 alpha=0.5, beta=0.5, gama=0, delta=0,
                 verbose=-1,
                 time_norm=None,
                 log_folder_name=None,
                 class_balance_mutation=False,
                 mutation_factor=0.5,
                 uniform_crossover=True,
                 n_jobs=1, 
                 patience=None,
                 early_stop=None,
                 sampling_start=0,
                 mutation_size_neighborhood=20,
                 mutation_percentage_change=0.1,
                 retrain_all_data=None,
                 search_space_config=None,
                 flaml_ms=False,
                 seed=42):
        """
        Initialization of the class

        Parameters:
        ----------
        task : str
            Type of task to do. It can be 'classification' or 'regression'

        metric : str
            Name of the function to use as metric. It can be 'mcc', 'f1', 'accuracy', 'precision', 'recall' or 'roc_auc'. Otherwise, it will raise an error

        validation_size : float
            Percentage of validation data to use on the internal division between train and validation

        n_iterations : integer
            Number of iterations to do of the optimization process. It will be ignored if the time_budget is not None

        binary_sampling_component : bool
            To use the binary sampling component or not

        automatic_data_optimization : bool
            To use the automatic data optimization or not. If False, it will use according use_sampling, use_feature_selection and use_data_augmentation

        use_sampling: bool
            Boolean to indicate of the sampling should be applied or not

        use_feature_selection: bool
            Boolean to indicate of the feature selection should be applied or not

        prob_mutation : float
            Probability of mutation

        prob_mutation_model : float
            Probability of mutation of the model

        prob_crossover : float
            Probability of crossover

        tournament_size : integer
            Tournament size

        elitism_size : integer
            Number of the best individuals to keep from one generation to the next

        population_size : integer
            Size of the population

        alpha, beta, gama, delta: float
            Weights of the different components of the fitness function

        verbose : integer
            Tells which populations should be saves along the optimisation. -1 = save all, 0 = save initial population and final one,
            Other integer says the step of generations to save. 

        time_norm : integer
            Normalization of the time component of the fitness. Maximum value accepted

        log_folder_name : str
            Directory where to save the information / logs

        class_balance_mutation : bool
            To use balance mutation or not

        mutation_factor : float
            Mutation factor to add to the probabilities fot the class balance mutation

        uniform_crossover : bool
            To use uniform crossover (True) or point crossover (false)

        n_jobs : integer
            Number of parallel workers to use

        patience : integer or None
            Number of generations to wait until restart the population. If None, it will not restart

        early_stop : integer or None
            Number of generations to wait until finish the optimization process without improvement. If None, it will not finish
        
        sampling_start : integer
            Iteration where to start the sampling. By default, it starts in the first iteration

        mutation_size_neighborhood : integer
            Size of the neighborhood to change in the mutation (integer representation)

        mutation_percentage_change : float
            Percentage of the dimension to change in the mutation (integer representation)

        retrain_all_data : str or None
            To retrain with all samples ('samples'), with all features ('features'), with all samples and features ('all') or None

        seed : integer
            Seed to use in the process

        Returns:
        -------
            -
        """
        super().__init__()
        self.seed = seed
        # check the type of task
        assert task in ['classification', 'regression'], 'Task must be classification or regression'
        self.task = task
        self.metric_name = metric
        self.metric = error_metric_function(metric, self.task)
        self.validation_size = validation_size
        self.n_iterations = n_iterations
        self.binary_sampling_component = binary_sampling_component
        self.automatic_data_optimization = automatic_data_optimization
        self.use_sampling = use_sampling
        self.use_feature_selection = use_feature_selection
        self.use_data_augmentation = use_data_augmentation
        self.sampling_start = sampling_start
        self.prob_mutation = prob_mutation
        self.prob_mutation_model = prob_mutation_model
        self.prob_crossover = prob_crossover
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.population_size = population_size
        self.time_budget = time_budget  # seconds
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.delta = delta
        self.verbose = verbose
        self.time_norm = time_norm
        self.class_balance_mutation = class_balance_mutation
        self.uniform_crossover = uniform_crossover
        self.mutation_factor = mutation_factor
        self.log_folder_name = log_folder_name
        self.n_jobs = n_jobs
        self.patience = patience
        self.early_stop = early_stop
        self.mutation_size_neighborhood = mutation_size_neighborhood
        self.mutation_percentage_change = mutation_percentage_change
        self.retrain_all_data = retrain_all_data
        self.flaml_ms = flaml_ms

        if self.log_folder_name == None:
            self.log_folder_name = os.path.join(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        if self.log_folder_name and os.path.exists(self.log_folder_name) == False:
            os.makedirs(self.log_folder_name)
            if self.flaml_ms:
                os.makedirs(os.path.join(self.log_folder_name, 'flaml_integration'))
        self.error_search = False

        if isinstance(search_space_config, dict):
            # received a config search spave
            self.config_models = search_space_config
        elif isinstance(search_space_config, str): 
            # path to the config file
            with open(os.path.join('edca', 'configs', search_space_config), 'r') as file:
                self.config_models = json.load(file)
        else:
            # setup models is None
            with open(os.path.join('edca', 'configs', f'{self.task}_models.json'), 'r') as file:
                self.config_models = json.load(file)

    def _save(self):
        """ Saves the configuration used, the best individuals found over the iterations and the class distributions of the sampling data"""
        # save best individuals and fitness components
        self.search_algo.bests_info['config'] = [json.dumps(config, cls=NpEncoder) for config in self.search_algo.bests_info['config']]
        
        bests = pd.DataFrame(self.search_algo.bests_info)
        bests.index = pd.Series(bests.index, name='Iteration') + 1
        bests.to_csv(os.path.join(self.log_folder_name, 'bests.csv'))

        # save target classes used in sampling / training
        self.internal_y_train.name = 'target_class'
        aux = self.internal_y_train.to_frame()
        aux.to_csv(os.path.join(self.log_folder_name, 'train_y.csv'))

        # save internal train and validation data
        aux = self.internal_x_train.copy()
        aux['target_class'] = self.internal_y_train
        aux.to_csv(os.path.join(self.log_folder_name, 'internal_train_data.csv'))

        aux = self.internal_x_val.copy()
        aux['target_class'] = self.internal_y_val
        aux.to_csv(os.path.join(self.log_folder_name, 'internal_val_data.csv'))

        if self.error_search == False:
            # save best data selected
            aux = self.pipeline_estimator.X_train.copy()
            aux['target_class'] = self.pipeline_estimator.y_train
            aux.to_csv(os.path.join(self.log_folder_name, 'best_data.csv'))

            # save best samples data
            aux_x, aux_y = self.pipeline_estimator.get_best_sample_data()
            aux_x['target_class'] = aux_y
            aux_x.to_csv(os.path.join(self.log_folder_name, 'best_sample_data.csv'))

        # save pipeline config
        with open(os.path.join(self.log_folder_name, 'pipeline_config.json'), 'w') as file:
            json.dump(self.pipeline_config, file, cls=NpEncoder, indent=2)

    def fit(self, X_train, y_train):
        """ Function fit the AutoML framework. It divides the data and uses it to find the best pipeline.
        In the end, it retrains and fits the best pipeline
        """
        # save dataset
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        # self.X_train.reset_index(drop=True, inplace=True)
        # self.y_train.reset_index(drop=True, inplace=True)

        # split data into train and validation to search the best config
        self.internal_x_train, self.internal_x_val, self.internal_y_train, self.internal_y_val = train_test_split(
            self.X_train, self.y_train, 
            shuffle = True, 
            stratify = y_train if self.task == 'classification' else None, 
            test_size = self.validation_size,
            random_state = self.seed
            )
        self.sampling_size = len(self.internal_x_train)
        self.fs_size = self.internal_x_train.shape[1]

        # analyse the dataset. Uses only the train data no analyse the results
        self.pipeline_config = dataset_analysis(self.internal_x_train)
        # add the sampling config
        self.pipeline_config['sampling'] = self.use_sampling
        self.pipeline_config['feature_selection'] = self.use_feature_selection
        self.pipeline_config['data_augmentation'] = self.use_data_augmentation
        self.pipeline_config['alpha'] = self.alpha
        self.pipeline_config['beta'] = self.beta
        self.pipeline_config['gama'] = self.gama
        self.pipeline_config['delta'] = self.delta
        self.pipeline_config['time_norm'] = self.time_norm
        self.pipeline_config['sample-start'] = self.sampling_start
        self.pipeline_config['automatic_data_optimization'] = self.automatic_data_optimization
        self.pipeline_config['sampling_size'] = self.sampling_size
        self.pipeline_config['fs_size'] = self.fs_size
        self.pipeline_config['task'] = self.task
        self.pipeline_config['seed'] = self.seed
        if self.flaml_ms:
            self.pipeline_config['flaml_ms'] = True
            self.pipeline_config['search_metric'] = self.metric_name
            self.pipeline_config['time_budget'] = self.time_budget
            self.pipeline_config['flaml_save_dir'] = os.path.join(self.log_folder_name, 'flaml_integration')

        # optimisation process
        self._search_algorithm(
                X_train=self.internal_x_train,
                X_val=self.internal_x_val,
                y_train=self.internal_y_train,
                y_val=self.internal_y_val)

        if self.search_algo.best_individual is None:
            # if an error occurs
            self.error_search = True
            self.best_individual = None
            self.final_individual = None
        else:

            self.best_individual = self.search_algo.best_individual.copy()
            ## add FLAML config of the best model
            if self.flaml_ms:
                self.best_individual['flaml_estimator'] = self.search_algo.best_fitness_params['flaml_estimator']
                self.best_individual['flaml_estimator_config'] = self.search_algo.best_fitness_params['flaml_estimator_config']
            
            # convert indices for general dataset
            if 'sample' in self.best_individual:
                # change best samples to the general dataset
                selected_samples_indices = list(self.internal_x_train.iloc[self.best_individual['sample']].index)
                numerical_indices = list(np.arange(len(self.X_train))[self.X_train.index.isin(selected_samples_indices)])
                self.best_individual['sample'] = numerical_indices

            self.final_individual = self.best_individual.copy()
            # check if we need to retrain with all data (samples and/or features)
            if self.retrain_all_data == 'samples':
                self.final_individual.pop('sample', None)
            if self.retrain_all_data == 'features':
                self.final_individual.pop('features', None)
            if self.retrain_all_data == 'all':
                self.final_individual.pop('sample', None)
                self.final_individual.pop('features', None)            

            self.pipeline_estimator = PipelineEstimator(
                individual_config=self.final_individual,
                pipeline_config=self.pipeline_config,
                individual_id='best_individual'
            )

            self.pipeline_estimator.fit(self.X_train, self.y_train)

            # calculated selected data
            if 'sample' in self.best_individual or 'features' in self.best_individual:
                self.selected_X_train, self.selected_y_train, _ = get_selected_data(
                    self.X_train, self.y_train, self.best_individual)
            else:
                self.selected_X_train = self.pipeline_estimator.X_train
                self.selected_y_train = self.pipeline_estimator.y_train

            # saves the best individual found and other configs
            if self.log_folder_name is not None:
                self._save()
        return self

    def _search_algorithm(self, X_train, X_val, y_train, y_val):
        """ Optimisation process to found the best ML pipeline"""

        # select sample mutation
        if self.class_balance_mutation:
            class_balance_func = sample_class_balance_mutation(
                y_train, self.mutation_factor)
        else:
            class_balance_func = None

        sampling_mutation_operator = mutation_sampling_component(
            prob_mutation=self.prob_mutation,
            dimension=self.sampling_size,
            binary_representation=self.binary_sampling_component,
            size_neighborhood=self.mutation_size_neighborhood,
            max_number_changes=max(1, int(self.sampling_size * self.mutation_percentage_change)),
            class_balance_func=class_balance_func
        )

        fs_mutation_operator = mutation_sampling_component(
            prob_mutation=self.prob_mutation,
            dimension=self.fs_size,
            binary_representation=self.binary_sampling_component,
            size_neighborhood=self.mutation_size_neighborhood,
            max_number_changes=max(1, int(self.fs_size * self.mutation_percentage_change)),
            class_balance_func=None

        )

        # select crossover operator
        if self.uniform_crossover:
            crossover_operator = uniform_crossover(
                binary_representation=self.binary_sampling_component)
        else:
            crossover_operator = points_crossover(
                binary_representation=self.binary_sampling_component)

        sampling_generator = generate_sampling_component(
            binary_representation=self.binary_sampling_component)

        mutation_operator = mutation_individuals(
            prob_mutation=self.prob_mutation,
            prob_mutation_model=self.prob_mutation_model,
            config=self.config_models,
            sample_mutation_operator=sampling_mutation_operator,
            fs_mutation_operator=fs_mutation_operator,
            pipeline_config=self.pipeline_config,
            data_generator = sampling_generator
        )

        # search the best pipeline
        self.search_algo = EvolutionarySearch(
            config_models=self.config_models,
            pipeline_config=self.pipeline_config,
            mutation_operator=mutation_operator,
            crossover_operator=crossover_operator,
            sampling_generator=sampling_generator,
            prob_mutation=self.prob_mutation,
            prob_mutation_model=self.prob_mutation_model,
            prob_crossover=self.prob_crossover,
            population_size=self.population_size,
            tournament_size=self.tournament_size,
            elitism=self.elitism_size,
            num_iterations=self.n_iterations,
            time_budget=self.time_budget,
            filepath=self.log_folder_name,
            components={
                'alpha': self.alpha,
                'beta': self.beta,
                'gama': self.gama
            },
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            fitness_metric=self.metric,
            n_jobs=self.n_jobs,
            patience=self.patience,
            early_stop=self.early_stop,
            verbose=self.verbose
        )

        self.search_algo.evolutionary_algorithm()


    def predict(self, X):
        """ Predicts the test set using the best ML pipeline found during the optimisation process"""
        if self.error_search:
            return None
        preds = self.pipeline_estimator.predict(X)
        return preds

    def predict_proba(self, X):
        """ Predicts the probability of test sample with the best ML pipeline found """
        if self.error_search:
            return None
        preds_proba = self.pipeline_estimator.predict_proba(X)
        return preds_proba

    def get_best_individual(self):
        """ Returns the best individual"""
        return self.best_individual
    
    def get_final_data_size(self):
        """ Calculates and returns the data size used on the final pipeline"""
        return self.pipeline_estimator.X_train.size

    def get_final_data_shape(self):
        """ Returns the shape of the final data used (the selected data or the size the retrained pipeline when we retrain with all data - samples and/or features)"""
        return self.pipeline_estimator.X_train.shape
    
    def get_final_data(self):
        """ Returns the final data used on the final pipeline (with retraining or not with all the samples and/or feature)"""
        return self.pipeline_estimator.X_train, self.pipeline_estimator.y_train
    
    def get_selected_data_size(self):
        if self.retrain_all_data is None:
            return self.get_final_data_size()
        return self.selected_X_train.size
    
    def get_selected_data_shape(self):
        if self.retrain_all_data is None:
            return self.get_final_data_shape()
        return self.selected_X_train.shape
    
    def get_selected_data(self):
        if self.retrain_all_data is None:
            return self.get_final_data()
        return self.selected_X_train, self.selected_y_train
    
    

