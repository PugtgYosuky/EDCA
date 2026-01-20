import numpy as np
from edca.model import *
import logging
from edca.fitness import *
import os
from edca.ea import *
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import json
from edca.encoder import NpEncoder
from edca.fitness import individual_fitness
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

WORST_FITNESS = 1
VERBOSE_SAVE_ALL = -1

class TimeBudgetExceeded(Exception):
    pass

def sort_dict(d):
    aux = {}
    for key, item in d.items():
        if isinstance(item, dict):
            aux[key] = sort_dict(item)
        else:
            aux[key] = item
    return aux

class EvolutionarySearch:
    """ Aplies the evolutionary algorithm to find the best ML pipeline """

    def __init__(self,
            config_models,
            pipeline_config,
            crossover_operator,
            mutation_operator,
            sampling_generator,
            X_train,
            y_train,
            X_val,
            y_val,
            fitness_metric,
            prob_mutation=0.3,
            prob_mutation_model=0.5,
            prob_crossover=0.7,
            population_size=10,
            tournament_size=3,
            elitism=1,
            num_iterations=100,
            time_budget=-1,
            filepath='',
            n_jobs=5,
            patience=None,
            early_stop=None,
            verbose=-1,
            seed=42
            ):
        """
        Initialization of the class

        Parameters:
        ----------
        config_models : dict
            Search space of each type of gene

        pipeline_config : dict
            Data characteristics to generate the individuals

        crossover_operator : function
            Crossover operator to apply to the individuals

        mutation_operator : function
            Mutation operator to apply to the individuals

        sampling_generator : function
            Function to generate the sampling component of the individuals

        sampling_size : int
            Size of the sampling gene

        fs_size : int
            Size of the feature selection gene

        X_train : pandas.DataFrame
            Training data to train the individuals

        y_train : pandas.Series
            Training target to train the individuals

        X_val : pandas.DataFrame
            Validation data to evaluate the individuals

        y_val : pandas.Series
            Validation target to evaluate the individuals

        fitness_metric : function
            Metric to evaluate the individuals

        prob_mutation : float
            Mutation probability

        prob_mutation_model : float
            Mutation probability of the model component

        prob_crossover : float
            Crossover probability

        population_size : integer
            Size of the EA population

        tournament_size : integer
            Size of the tournament to select the best individuals

        elitism : integer
            Elitism size

        num_iterations : integer
            Number of generation todo of the EA. It will be ignored if the time_budget is defined

        time_budget : integer
            Time budget (in seconds) to use for the optimization process

        filepath : string
            Directory where to store the logs

        n_jobs : integer
            Number of parallel workers to apply

        patience : integer or None
            Number of generations to wait until restart the population. If None, it will not restart

        early_stop: integer or None
            Number of generations to wait until finish the search without improvement. If None, it will not finish
        
        verbose : integer
            Which generations to save the configs

        seed : integer
            Seed to reproduce the results
        Returns:
        -------
            -
        """
        # initialize variables
        self.prob_mutation = prob_mutation
        self.prob_mutation_model = prob_mutation_model
        self.prob_crossover = prob_crossover
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.config_models = config_models
        self.pipeline_config = pipeline_config
        self.num_iterations = num_iterations
        self.time_budget = time_budget
        self.iteration = 0
        self.mutation_operator = mutation_operator
        self.crossover_operator = crossover_operator
        self.sampling_generator = sampling_generator
        # to evaluate the individuals
        self.fitness_metric = fitness_metric
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.n_jobs = n_jobs
        self.patience = patience
        self.early_stop = early_stop
        self.verbose = verbose
        self.number_evaluated_individuals = 0

        # whither to use parallel or sequential search
        if self.n_jobs == 1:
            self._evaluate_population = self._evaluate_population_sequential
        else:
            self._evaluate_population = self._evaluate_population_parallel

        # setup save variables
        self.best_fit= None
        self.best_individual = None
        self.best_fitness_params = None

        self.bests_info = {}

        self.filepath = filepath

        # save individuals fitness to avoid retraining
        self._individuals_fitness_df = pd.DataFrame(
            columns=[
                'config',
                'fitness',
                'search_metric',
                'train_percentage',
                'time_cpu',
                'samples_percentage',
                'features_percentage',
                'balance_metric',
                'proportion_per_class']
        )

        # file to save populations
        self.pops_path = os.path.join(filepath, 'population')
        os.makedirs(self.pops_path)
        # setup log
        self._logger = logging.getLogger()
        self._logger.handlers = []
        self._logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s: %(levelname)-8s %(message)s')
        if filepath:
            filename = os.path.join(filepath, 'logger.log')
            file_handler = logging.FileHandler(filename)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        self._logger.addHandler(console)

    def save_evaluated_individuals(self):
        """ Saves the evaluated individuals """
        self._individuals_fitness_df.to_csv(os.path.join(self.filepath,'evaluated_individuals.csv'))

    def _save_population(self, filename):
        """ Saves the population of one generation"""
        population = []
        for _, fitness_params in self.population:
            population.append(fitness_params)
        df = pd.DataFrame(population)
        df.insert(0, 'individual', list(range(1, len(self.population) + 1)))
        df.to_csv(os.path.join(self.pops_path, f'Population_generation_{filename}.csv'), index=False)

        # save individuals config
        with open(os.path.join(self.pops_path, f'Populations_config_generation_{filename}.txt'), 'w') as file:
            for i in range(len(self.population)):
                aux = self.population[i][0].copy()
                aux['individual_id'] = self.population[i][1]['individual_id']
                file.write(json.dumps(aux,cls=NpEncoder) + "\n")

    def _check_time_limit(self):
        """ Tests it it as reached the limit time budget """
        if self.time_budget != -1:
            seconds_elapsed = (datetime.now() - self._start_datetime).total_seconds()

            if seconds_elapsed >= self.time_budget:
                raise TimeBudgetExceeded(
                    "{:.2f} seconds have elapsed. It will close down.".format(seconds_elapsed))

    def _add_individual_to_evaluated(self, string_individual, fitness_params):
        """ Stores the evaluated individual and its results to avoid repeating evaluations"""
        aux = {'config' : [string_individual]}
        aux_config = {**aux, **fitness_params}
        self._individuals_fitness_df = pd.concat([self._individuals_fitness_df,pd.DataFrame(aux_config)], axis=0,ignore_index=True)

    def _evaluate_population_parallel(self, population):
        """ Evaluates the population in parallel based on the number of workers"""
        # sort individuals config
        population = sort_individuals_configs(population)
        
        self.population_evaluated = []
        evaluated_individuals = {}  # to store the individuals evaluated
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            try:
                # iterate over the population and start processes
                for (individual_config, fitness_params) in population:
                    string_individual = str(individual_config)
                    # search for individual in the pool evaluated
                    equal_indiv = self._individuals_fitness_df.loc[
                        self._individuals_fitness_df.config == string_individual]
                    if len(equal_indiv) == 0:
                        # new individual
                        if string_individual not in evaluated_individuals:
                            # individual is not being tested yet
                            self.number_evaluated_individuals += 1
                            thread = executor.submit(individual_fitness,
                                self.X_train,
                                self.X_val,
                                self.y_train,
                                self.y_val,
                                self.pipeline_config,
                                self.fitness_metric,
                                individual_config,
                                self.number_evaluated_individuals
                            )
                            
                            evaluated_individuals[string_individual] = {
                                'thread': thread,
                                'individual': individual_config,
                                'count': 1
                            }
                        else:
                            # individual already being tested --> increment
                            evaluated_individuals[string_individual]['count'] = evaluated_individuals[string_individual]['count'] + 1
                    else:
                        # already tested individual
                        fitness_params = get_individual_params(equal_indiv)
                        self.population_evaluated.append((individual_config.copy(), fitness_params.copy()))
                    self._check_time_limit()

                # iterate over processes evaluated by the threadpoolexecuter
                string_individuals = list(evaluated_individuals.keys())
                for string_individual in string_individuals:
                    info = evaluated_individuals.pop(string_individual)
                    fitness_params = info['thread'].result()
                    for _ in range(info['count']):
                        self.population_evaluated.append((info['individual'], fitness_params))
                        self._add_individual_to_evaluated(string_individual, fitness_params)
                    self._check_time_limit()

            except TimeBudgetExceeded as e:
                # cancel all future threads from running
                string_individuals = list(evaluated_individuals.keys())
                for string_individual in string_individuals:
                    info = evaluated_individuals[string_individual]
                    if not info['thread'].running():
                        _ = evaluated_individuals.pop(string_individual)
                        info['thread'].cancel()

                # wait and get results from running threads
                for string_individual, info in evaluated_individuals.items():
                    fitness_params = info['thread'].result()
                    # save evaluated individuals in the population
                    for _ in range(info['count']):
                        self.population_evaluated.append((info['individual'], fitness_params))
                    self._add_individual_to_evaluated(string_individual, fitness_params=fitness_params)

                # raise the exception
                raise TimeBudgetExceeded(e)
            except KeyboardInterrupt as e:
                # terminate all the process if ctrl-c pressed
                executor.shutdown(wait=False)
                # raise the exception
                raise KeyboardInterrupt(e)

        return sort_population(self.population_evaluated)

    def _evaluate_population_sequential(self, population):
        """ Evaluates the populations based on the fitness function sequentially"""
        ## sort individuals config
        population = sort_individuals_configs(population)

        self.population_evaluated = []
        for (individual_config, fitness_params) in population:
            # convert individual to string
            string_individual = str(individual_config)
            # search for individual in the pool evaluated
            equal_indiv = self._individuals_fitness_df.loc[
                self._individuals_fitness_df.config == string_individual]
            if len(equal_indiv) == 0:
                self.number_evaluated_individuals += 1
                # the individual was not found
                fitness_params = individual_fitness(
                    X_train=self.X_train,
                    X_val=self.X_val,
                    y_train=self.y_train,
                    y_val=self.y_val,
                    metric=self.fitness_metric,
                    pipeline_config=self.pipeline_config,
                    individual=individual_config,
                    individual_id = self.number_evaluated_individuals)
                # add individual to pool of evaluated individuals
                self._add_individual_to_evaluated(string_individual, fitness_params)
            else:
                fitness_params = get_individual_params(equal_indiv)
            # add to the population
            self.population_evaluated.append((individual_config.copy(), fitness_params.copy()))
            self._check_time_limit()
        # sort population
        return sort_population(self.population_evaluated)

    def _select_survivals(self, offspring):
        """ Select the survivals, based on the elitism size and the offspring"""
        offspring = self._evaluate_population(offspring.copy())
        # applies elitism
        new_pop = self.population[:self.elitism] + offspring[:-self.elitism]
        return sort_population(new_pop)

    def evolutionary_algorithm(self):
        self.counter_repeated = 0
        self.counter_no_improvement = 0
        """ Evolutionary algorithm """
        self._logger.info('Evolutionary Search')

        # start time counter
        self._start_datetime = datetime.now()
        self.pipeline_config['start_datetime'] = time.time()

        # create initial population
        self._logger.info('Create Initial Population')
        try:
            self.population = generate_population(
                pop_size=self.population_size,
                config=self.config_models,
                pipeline_config=self.pipeline_config,
                sampling_generator=self.sampling_generator
            )
        except KeyboardInterrupt:
            raise KeyboardInterrupt('Ctrl-C pressed')
        except TimeBudgetExceeded as error:
            self._logger.info('No time to evaluate the initial population')
            raise TimeBudgetExceeded(
                'No time to evaluate the initial Population')

        self._logger.info('Evaluate Initial Population')
        try:
            self.population = deepcopy(self._evaluate_population(self.population))
            self.best_fit= self.population[0][1]['fitness']
            self.best_individual = deepcopy(self.population[0][0])
            self.best_fitness_params = deepcopy(self.population[0][1])
        except KeyboardInterrupt:
            raise KeyboardInterrupt('Ctrl-C pressed')
        except TimeBudgetExceeded as e:
            self.population_evaluated = sort_population(self.population_evaluated)
            self.population = deepcopy(self.population_evaluated)
            self.best_individual = deepcopy(self.population[0][0])
            self.best_fit= self.population[0][1]['fitness']
            self.best_fitness_params = deepcopy(self.population[0][1])
            self._save_info()
            self._save_population('Initial_Incomplete')
            self._logger.info(
                f'Time ended. Only {len(self.population)} were evaluated from the initial population')
            return
        self._save_info()
        self._save_population('Initial')
        
        try:
            self._logger.info('Start Search for the best pipeline')
            while (self.time_budget != -1 or self.iteration < self.num_iterations - 1) and self.counter_no_improvement != self.early_stop:

                self._check_time_limit()

                # create offspring by tournament
                offspring = []
                for _ in range(self.population_size):
                    selected_indivs = random.choices(
                        self.population, 
                        k=self.tournament_size
                    )
                    selected_indiv = tournament(selected_indivs, sort_function=sort_population)
                    offspring.append(selected_indiv)
                self._check_time_limit()

                # add crossover
                for i in range(self.population_size - 1):
                    if np.random.random() < self.prob_crossover:
                        a, b = self.crossover_operator(offspring[i][0], offspring[i + 1][0])
                        offspring[i] = (a, {'fitness': WORST_FITNESS})
                        offspring[i + 1] = (b, {'fitness': WORST_FITNESS})
                self._check_time_limit()

                # add mutation
                for i in range(self.population_size):
                    if np.random.random() < self.prob_mutation:
                        offspring[i] = (self.mutation_operator(offspring[i][0]), {'fitness': WORST_FITNESS})
                self._check_time_limit()

                # select survivals
                self.population = self._select_survivals(offspring=offspring)
                # get best fitness
                best_info = self.population[0][1]

                # check if its the same best
                if self.best_fit== best_info['fitness']:
                    self.counter_repeated += 1  # increment counter
                    self.counter_no_improvement += 1
                else:
                    self.counter_repeated = 0  # reinitialize counter
                    self.counter_no_improvement = 0
                    self.best_fit= best_info['fitness']
                self.best_individual = deepcopy(self.population[0][0])
                self.best_fitness_params = deepcopy(self.population[0][1])

                # update logger and save population
                self.iteration += 1
                self._logger.info(
                    f"Iteration {self.iteration} >>> Fitness: {best_info['fitness']:.3f} - Data%: {best_info['train_percentage']:.3f} - Metric: {best_info['search_metric']:.3f} - CPU Time: {best_info['time_cpu']:.3f} - S%: {best_info['samples_percentage']:.3f} - F%: {best_info['features_percentage']:.3f} - CDD: {best_info['balance_metric']:.3f}")
                
                # save best information
                self._save_info()

                # save population
                if self.verbose == VERBOSE_SAVE_ALL or (self.verbose != 0 and self.iteration % self.verbose == 0):
                    self._save_population(self.iteration)

                # check if it had achieved the maximum patience - restarts the population
                if self.counter_repeated == self.patience:
                    self._logger.info(
                        f'Iteration {self.iteration} >>> Restart of the population')
                    self.counter_repeated = 0  # reinitialize the counter
                    # generate new population
                    new_population = generate_population(
                        # generate only the new individuals
                        pop_size=self.population_size,
                        config=self.config_models,
                        pipeline_config=self.pipeline_config,
                        sampling_generator=self.sampling_generator
                    )
                    # keep the best
                    # save the best individuals + the new pop
                    self.population = self._select_survivals(new_population)
                self._check_time_limit()

                # add the sampling component after X iterations when configured
                # it does not re-evaluates the population at this stage, only
                # in the next generation
                if self.pipeline_config['automatic_data_optimization'] == False and self.pipeline_config['sampling'] and self.pipeline_config['sample-start'] == self.iteration:
                    new_pop = []
                    self._logger.info(
                        f'Iteration {self.iteration} >>> Add Sampling component'
                    )
                    for (indiv,fitness_params) in self.population:
                        indiv['sample'] = self.sampling_generator(size=self.pipeline_config['sampling_size'])
                        new_pop.append((indiv, fitness_params))
                    self.population = new_pop
                self._check_time_limit()
        except KeyboardInterrupt:
            raise KeyboardInterrupt('Ctrl-C pressed')
        except TimeBudgetExceeded as e:
            pass

        # save population
        if self.verbose == 0 or self.iteration % self.verbose != 0: # if not already saved
            self._save_population(self.iteration)

        self._logger.info(
                f'Search Ended after {self.iteration} iterations with a time of {(datetime.now()-self._start_datetime).total_seconds()} seconds')

        self.save_evaluated_individuals()
        if self.best_fit == WORST_FITNESS:
            self.best_individual = None
            self.best_fitness_params = None
            self._logger.info('No pipeline was found. All pipelines resulted in error')
        # self.save_plots()

    def _save_info(self):
        """ Saves the information about the generation """
        # initial iteration values
        if 'average_fitness' not in self.bests_info:
            self.bests_info['average_fitness'] = []
            self.bests_info['config'] = []
            for key in self.population[0][1].keys():
                self.bests_info[key] = []
        # update values
        self.bests_info['average_fitness'].append(calculate_average_fitness(self.population))
        self.bests_info['config'].append(self.population[0][0].copy())
        for key, value in self.population[0][1].items():
            self.bests_info[key].append(value)

    def save_plots(self):
        """ Makes plots about the evolution over the generations """
        plt.figure(figsize=(15, 10))
        plt.plot(self.bests_info['fitness'], label='Bests')
        plt.plot(self.bests_info['average_fitness'], label='Average')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.savefig(os.path.join(self.filepath, 'fitness_evolution.png'))

        plt.figure(figsize=(15, 10))
        plt.plot(self.bests_info['fitness'], label='Bests')
        plt.plot(self.bests_info['average_fitness'], label='Average')
        plt.xlim([10, len(self.bests_info['fitness'])])
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.savefig(os.path.join(self.filepath, 'zoom_fitness_evolution.png'))

        plt.figure(figsize=(15, 10))
        plt.plot(self.bests_info['search_metric'], label='Search Metric')
        plt.plot(self.bests_info['time_cpu'], label='CPU Time')
        plt.plot(self.bests_info['train_percentage'], label='Train Percentage')
        plt.plot(self.bests_info['balance_metric'], label='balance')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Component Value')
        plt.savefig(os.path.join(self.filepath, 'components_evolution.png'))
        
    def get_number_iterations(self):
        """ returns the number of generations made"""
        return self.iteration

    def get_number_pipelines_tested(self):
        """ Returns the number of pipelines tested. Equals to the num generations * population size """
        return self.iteration * len(self.population) + len(self.population_evaluated)
    


    
def sort_population(population):
        population.sort(key=lambda x: x[1]['fitness'], reverse=False)
        return population

def get_individual_params(equal_indiv):
    return equal_indiv.to_dict(orient='records')[0]

def sort_individuals_configs(population):
    population = deepcopy(population)
    for index, (individual_config, fitness_params) in enumerate(population):
        population[index] = (sort_dict(individual_config), fitness_params)
    return population
