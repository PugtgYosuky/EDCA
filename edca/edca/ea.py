import numpy as np
import pandas as pd
from operator import itemgetter
from edca.model import *
import random
from copy import deepcopy


# CROSSOVER
def uniform_crossover(prob_crossover=0.5, binary_representation=True):
    """
    Uniform crossover operator

    Applies uniform crossover to the sampling component of the individuals

    Parameters:
    ----------
    prob_crossover : float
        Probability of crossover

    binary_representation : bool
        If the representation is binary or not

    Returns:
    -------
    function
        to crossover the sampling component of the individuals

    """
    sample_crossover = crossover_sampling_component(uniform_crossover=True, binary_representation=binary_representation)

    def crossover(indiv_a, indiv_b):
        """
        Uniform crossover of two individuals

        Applies uniform crossover to two individuals, resulting in two new offsprings.
        It applies uniform crossover at the sampling component of the individuals and
        also al the level of the other components.

        Parameters:
        ----------
        indiv_a : dict
            Parent A

        indiv_b : dict
            Parent B

        Returns:
        -------
        dict, dict
            The two new offsprings with some gene swapped between the two given parents

        """
        offspring_a = indiv_a.copy()
        offspring_b = indiv_b.copy()
        # only the common keys are considered
        keys = list(set(offspring_a.keys()) & set(offspring_b.keys()))
        for i in keys:
            if i == 'sample' or i == 'features':
                # applies uniform crossover to change the sampling of the
                # individuals
                sample_a, sample_b = sample_crossover(indiv_a[i], indiv_b[i])
                offspring_a[i] = sample_b
                offspring_b[i] = sample_a
            if np.random.random() < prob_crossover:
                offspring_a[i] = indiv_b[i]
                offspring_b[i] = indiv_a[i]

        return offspring_a, offspring_b
    return crossover


def points_crossover(binary_representation=True):
    
    def crossover(indiv_a, indiv_b):
        """
        Point-crossover

        Applies two-point crossover internally in the sampling component of the individuals
        and 1-point crossover to the other components of the individuals.

        Parameters:
        ----------
        indiv_a : dict
            Parent A

        indiv_b : dict
            Parent B

        Returns:
        -------
        list, list
            New two offsprings

        """
        offspring_a = indiv_a.copy()
        offspring_b = indiv_b.copy()
        sample_crossover = crossover_sampling_component(
            uniform_crossover=False, binary_representation=binary_representation)
        # apply crossover to the sample component
        if 'sample' in offspring_a and 'sample' in offspring_b:
            # apply two point crossover to the sampling component
            off_sample_a, off_sample_b = sample_crossover(
                offspring_a['sample'], offspring_b['sample'])

            offspring_a['sample'] = off_sample_a
            offspring_b['sample'] = off_sample_b

        if 'features' in offspring_a and 'features' in offspring_b:
            # apply two point crossover to the sampling component
            off_sample_a, off_sample_b = sample_crossover(
                offspring_a['features'], offspring_b['features'])

            offspring_a['features'] = off_sample_a
            offspring_b['features'] = off_sample_b

        # only the common keys are considered
        keys = list(set(offspring_a.keys()) & set(offspring_b.keys()))
        keys = [key for key in keys if key !=
                'sample' and key != 'features']

        # iterate over the other components of the individual
        point = np.random.randint(0, len(keys))
        for i, key in enumerate(keys):
            if i > point:
                # swap values
                offspring_a[key] = indiv_b[key]
                offspring_b[key] = indiv_a[key]

        return offspring_a, offspring_b
    return crossover

# MUTATION

def mutation_individuals(
        prob_mutation,
        prob_mutation_model,
        config,
        sample_mutation_operator,
        fs_mutation_operator,
        pipeline_config,
        data_generator):
    """
    Mutation operator

    Mutates a given individual. It uses different mutation functions for the different genes of the individual
    Only mutates 1 gene per individual

    Parameters:
    ----------
    prob_mutation : float
        Probability of mutation

    config : dict
        Configuration file with the search space of the different genes

    sample_mutation_operator : function
        Mutation operator to use in the sampling gene

    fs_mutation_operator : function
        Mutation operator to use in the feature selection gene

    pipeline_config : dict
        Data characteristics

    Returns:
    -------
    function
        to mutate a given individual

    """
    """ mutates individuals """
    # get genes different types of mutations according to its configuration
    numerical_imputer_mutation = models_mutations(config['numerical-imputer'])
    categorical_imputer_mutation = models_mutations(config['categorical-imputer'])
    encoder_mutation = models_mutations(config['encoder'])
    model_mutation = models_mutations(config['model'])
    augmentation_mutation = models_mutations(config['data_augmentation'])

    # add the options to mutate the data according to the pipeline configuration (user preferences)
    data_options = []
    if pipeline_config['sampling']:
        data_options.append('sample')
    if pipeline_config['feature_selection']:
        data_options.append('features')
    if pipeline_config['data_augmentation']:
        data_options.append('data_augmentation')

    def mutation(individual):
        """
        Mutation operator to each individual

        Receives an individual and selects a gene from it to mutate. Based on the gene selected, it applies different types of mutation

        Parameters:
        ----------
        individual : dict
            Individual to mutate

        Returns:
        -------
        dict
            Mutates individual
        """
        new_individual = individual.copy()
        # selects only one gene to mutate per individual
        options = set(list(new_individual.keys())) | set(data_options)
        key = random.choice(list(options))

        if key == 'sample':
            if key in individual:
                # sample mutation
                if pipeline_config['automatic_data_optimization'] and np.random.random() < 0.5:
                    # if automatic data optimization is enabled, 
                    #the sample gene has a 50% chance of being removed
                    new_individual.pop(key)
                else:
                    new_individual[key] = sample_mutation_operator(new_individual[key])
            else:
                new_individual[key] = data_generator(pipeline_config['sampling_size'])

        if key == 'features':
            if key in new_individual:
                # sample mutation
                if pipeline_config['automatic_data_optimization'] and np.random.random() < 0.5:
                    # if automatic data optimization is enabled, 
                    #the features gene has a 50% chance of being removed
                    new_individual.pop(key)
                else:
                    new_individual[key] = fs_mutation_operator(new_individual[key])
            else:
                new_individual[key] = data_generator(pipeline_config['fs_size'])
        if key == 'data_augmentation':
            if key in new_individual:
                # sample mutation
                if pipeline_config['automatic_data_optimization'] and np.random.random() < 0.5:
                    # if automatic data optimization is enabled, 
                    #the data augmentation gene has a 50% chance of being removed
                    new_individual.pop(key)
                else:
                    new_individual[key] = augmentation_mutation(
                        model_config=new_individual[key],
                        prob_model_mutation=prob_mutation_model,
                        prob_mutation=prob_mutation    
                    )
            else:
                individual['data_augmentation'] = generate_model_code(config['data_augmentation'])
        if key == 'numerical-imputer':
            # numerical imputer mutation
            new_individual[key] = numerical_imputer_mutation(
                model_config=new_individual[key],
                prob_model_mutation=prob_mutation_model,
                prob_mutation=prob_mutation
            )
        if key == 'categorical-imputer':
            # categorical imputer mutation
            new_individual[key] = categorical_imputer_mutation(
                model_config=new_individual[key],
                prob_model_mutation=prob_mutation_model,
                prob_mutation=prob_mutation
            )
        if key == 'encoder':
            # encoder mutation
            new_individual[key] = encoder_mutation(
                model_config=new_individual[key],
                prob_model_mutation=prob_mutation_model,
                prob_mutation=prob_mutation
            )
        if key == 'scaler':
            # scaler mutation
            new_individual[key] = scalers_mutation(individual[key])
        if key == 'model':
            # model mutation
            new_individual[key] = model_mutation(
                model_config=new_individual[key],
                prob_model_mutation=prob_mutation_model,
                prob_mutation=prob_mutation
            )
        return new_individual

    return mutation

# GENERATE INDIVIDUALS


def generate_individual(config, pipeline_config, sampling_generator):
    """
    Generates an individual

    Generates a new individual based on the pipeline config received (data characteristics).

    Parameters:
    ----------
    sampling_size : int
        Size of the sampling component

    fs_size : int
        Size of the feature selection component

    config : dict
        Configuration file with the search space of the different genes

    pipeline_config : dict
        Characteristics od the data to generate the new individual

    Returns:
    -------
    dict
        Individual created
    """
    individual = {}

    if pipeline_config['automatic_data_optimization']:
        # each gene has a 50% chance of being included in the individual
        if pipeline_config['sampling'] and np.random.random() < 0.5:
            individual['sample'] = sampling_generator(pipeline_config['sampling_size'])
        if pipeline_config['feature_selection'] and np.random.random() < 0.5:
            individual['features'] = sampling_generator(pipeline_config['fs_size'])
        if pipeline_config['data_augmentation'] and np.random.random() < 0.5:
            individual['data_augmentation'] = generate_model_code(
                config['data_augmentation']
            )
    else:
        # manual defined if it should use the automatic data optimization
        if pipeline_config['sampling'] and pipeline_config['sample-start'] == 0:
            individual['sample'] = sampling_generator(pipeline_config['sampling_size'])

        if pipeline_config['feature_selection']:
            individual['features'] = sampling_generator(pipeline_config['fs_size'])
        
        if pipeline_config['data_augmentation']:
            individual['data_augmentation'] = generate_model_code(
                config['data_augmentation']
            )

    if len(pipeline_config['numerical_with_nans']) > 0:
        individual['numerical-imputer'] = generate_model_code(
            config['numerical-imputer'])

    if len(pipeline_config['categorical_with_nans']) > 0 or len(
            pipeline_config['binary_with_nans']) > 0:
        individual['categorical-imputer'] = generate_model_code(
            config['categorical-imputer'])

    if len(pipeline_config['numerical_columns']) > 0:
        individual['scaler'] = generate_scaler_code()

    if len(pipeline_config['categorical_columns']) > 0:
        individual['encoder'] = generate_model_code(config['encoder'])

    if not pipeline_config.get('flaml_ms', False): 
        # if the flaml model selection is activated, it does not uses the model gene
        individual['model'] = generate_model_code(config['model'])
    return (individual, {'fitness' : 1})


def generate_population(pop_size, config, pipeline_config, sampling_generator):
    """
    Generates the population

    Creates each individual based on the data characteristics

    Parameters:
    ----------
    sampling_size : int
        Size of the sampling gene

    fs_size : int
        Size of the feature selection gene

    pop_size: int
        Size of the desired population

    config : dict
        Configuration file with the search space of the different genes

    pipeline_config : dict
        Data characteristics

    Returns:
    -------
    list
        EA population
    """
    return [generate_individual(config, pipeline_config, sampling_generator) for _ in range(pop_size)]


def tournament(individuals, sort_function):
    """
    Tournament selection

    Receives a list of individuals and selects the best one

    Parameters:
    ----------
    individuals : list
        List of individuals in the tournament

    Returns:
    -------
    dict
        Best individual
    """
    individuals = sort_function(individuals)
    return deepcopy(individuals[0])


def calculate_average_fitness(population):
    """
    Calculates the average fitness of the population

    Parameters:
    ----------
    population : list
        List with the individual in the population

    Returns:
    -------
    float
        Mean fitness of the population
    """
    return np.mean([indiv[1]['fitness'] for indiv in population])


def generate_sampling_component(binary_representation=True):
    """
    Generates a sampling component

    Generates a sampling component based on the dimension received

    Parameters:
    ----------
    dimension : int
        Size of the sampling component

    binary_representation : bool
        If the representation is binary or not

    Returns:
    -------
    list
        Sampling component
    """
    if binary_representation:
        return generate_sample_code
    return generate_integer_representation


def generate_sample_code(size):
    """ Generates the code for the sampling component with binary representation """
    return [np.random.choice([0, 1]) for _ in range(size)]


def generate_integer_representation(chromosome_length):
    """ Generates the code for the sampling component with integer representation"""
    size = np.random.randint(0, chromosome_length-1)
    values = list(range(chromosome_length))
    individual = list(np.random.choice(values, min(len(values), size), replace=False))
    return individual


def mutation_sampling_component(
        prob_mutation,
        dimension,
        binary_representation=True,
        size_neighborhood=10,
        max_number_changes=10,
        class_balance_func=None):
    """
    Mutation operator

    Mutates the sampling component of the individual. It uses different mutation functions for the different genes of the individual
    Only mutates 1 gene per individual

    Parameters:
    ----------
    prob_mutation : float
        Probability of mutation

    dimension : int
        Size of the sampling component

    binary_representation : bool
        If the representation is binary or not

    size_neighborhood : int
        Size of the neighborhood to change the gene

    max_number_changes : int
        Number of changes to apply in the mutation

    class_balance_func : function
        Function to apply class balance mutation

    Returns:
    -------
    function
        to mutate the sampling component of a given individual

    """
    if binary_representation:
        if class_balance_func:
            return class_balance_func
        return sample_bitflip_mutation(prob_mutation_model=prob_mutation)
    return int_mutation(size_neighborhood, dimension, max_number_changes)


def sample_bitflip_mutation(prob_mutation_model=0.5):
    """
    Bit flip mutation of the sampling component

    Applies a bit flip mutation to the sampling component of a given individual

    Parameters:
    ----------
    indiv : list
        Binary list with the sampling component


    Returns:
    -------
    return_type
        Description of what the function returns.

    Raises:
    -------
    ExceptionType
        Description of when and why this exception might be raised,
        if applicable.

    Notes:
    ------
    Any additional notes or considerations about the function.
    """
    def mutate(chromosome):
        cromo = chromosome.copy()
        for i in range(len(cromo)):
            if np.random.random() < prob_mutation_model:
                cromo[i] = 1 - cromo[i]
        return cromo
    return mutate


def sample_class_balance_mutation(target, factor=0.5):
    """
    Class balance mutation

    Applies mutation to the sampling gene based on the class balance.
    It goes throw all the position and based on the class balance of the sample selected and based on its value,
    it applies mutation or not.

    Parameters:
    ----------
    target : list (binary)
        Probability of mutation

    factor : float
        Value to add to the probabilities

    Returns:
    -------
    list
        Sampling gene mutated
    """

    # calculate proportion of each class in the target sequence
    proportions = target.value_counts() / len(target)
    proportions.name = 'prop'
    proportions = proportions.to_frame().T
    # calculate ideal proportion (if balance: ideal prop = 1 / num classes)
    ideal_prop = 1 / target.nunique()
    # calculate the mutation probability of each instance
    # prob = factor + ideal proportion (if balance) - class proportion
    prob_array = pd.Series(
        np.zeros(
            len(target)),
        name='prob_muta').astype(float)
    for c in target.unique():
        value = factor + ideal_prop - proportions[c].values[0]
        prob_array.loc[(target == c).values] = value

    def operator(indiv):
        for i in range(len(indiv)):
            # change if the value is 0 and as the previous probability or if it
            # is 1 and the probability is greater than the value
            if (indiv[i] == 0 and np.random.random() <= prob_array[i]) or (
                    indiv[i] == 1 and np.random.random() > prob_array[i]):
                indiv[i] = 1 - indiv[i]
        return indiv
    return operator


def int_mutation(size_neighborhood, dimension, max_number_changes):
    mutation_options = [
        int_change_mutation(
            size_neighborhood=size_neighborhood,
            dimension=dimension,
            max_number_changes=max_number_changes
        ),
        int_delete_mutation(max_number_changes=max_number_changes),
        int_add_mutation(dimension=dimension, max_number_changes=max_number_changes)
    ]

    def mutate(chromosome):
        # choose a random mutation operator - change, delete or add
        mutation_operator = np.random.choice(mutation_options)
        return mutation_operator(chromosome)
    return mutate


def int_change_mutation(size_neighborhood, dimension, max_number_changes):
    """ Mutate a random gene in the chromosome"""
    def change_mutate(chromosome):
        new_chromosome = chromosome.copy()  # copy the chromosome
        number_changes = np.random.randint(1, max(2, max_number_changes))
        change_indexes = np.random.choice(
            list(range(len(new_chromosome))), size=min(number_changes, len(new_chromosome)), replace=False)  # choose a random index to change
        for change_index in change_indexes:
            value = new_chromosome[change_index]
            # choose a random delta
            delta = np.random.choice(
                list(range(-size_neighborhood, size_neighborhood)))
            value += delta
            # ensure the value is within the limits
            value = max(0, min(value, dimension - 1))
            if value not in new_chromosome:
                # add value to the new chromosome if it is not already in it
                new_chromosome[change_index] = value
        return new_chromosome
    return change_mutate


def int_delete_mutation(max_number_changes):
    def delete_mutate(chromosome):
        if len(chromosome) > 1:
            new_chromosome = chromosome.copy()
            number_changes = np.random.randint(1, max(2, max_number_changes))
            for _ in range(number_changes):
                if len(new_chromosome) == 1:
                    break
                delete_index = np.random.choice(
                    list(range(len(new_chromosome))))
                new_chromosome.pop(delete_index)
            return new_chromosome
        return chromosome.copy()
    return delete_mutate


def int_add_mutation(dimension, max_number_changes):
    def add_mutate(chromosome):
        new_chromosome = chromosome.copy()
        # create a set with all the possible values
        all_possible_values = set(list(range(dimension)))
        chromosome_set = set(chromosome)
        if len(all_possible_values) == len(chromosome):
            return new_chromosome
        # get the possible candidates to add
        possible_values = list(all_possible_values - chromosome_set)
        number_changes = np.random.randint(1, max(2, max_number_changes))
        new_genes = list(
            np.random.choice(
                possible_values,
                size=min(
                    number_changes,
                    len(possible_values)),
                replace=False))
        new_chromosome = new_chromosome + new_genes
        return new_chromosome
    return add_mutate


def crossover_sampling_component(
        uniform_crossover=True,
        binary_representation=True):
    """
    Crossover operator

    Applies crossover to the sampling component of the individuals

    Parameters:
    ----------
    uniform_crossover : bool
        If uniform crossover is applied or not

    binary_representation : bool
        If the representation is binary or not

    prob_crossover : float
        Probability of crossover

    Returns:
    -------
    function
        to crossover the sampling component of the individuals

    """
    if binary_representation:
        if uniform_crossover:
            return sample_uniform_crossover
        return sample_two_point_crossover
    return int_point_crossover


def sample_uniform_crossover(indiv_a_sample, indiv_b_sample):
    sample_a = indiv_a_sample.copy()
    sample_b = indiv_b_sample.copy()
    for i in range(len(indiv_a_sample)):
        if np.random.random() < 0.5:
            sample_a[i] = indiv_b_sample[i]
            sample_b[i] = indiv_a_sample[i]
    return sample_a, sample_b


def int_point_crossover(chromosome1, chromosome2):
    point_1 = np.random.randint(0, max(1, len(chromosome1) - 1))
    point_2 = np.random.randint(0, max(1, len(chromosome2) - 1))
    chromo1 = list(set(chromosome1[:point_1] + chromosome2[point_2:]))
    chromo2 = list(set(chromosome2[:point_2] + chromosome1[point_1:]))
    return chromo1, chromo2


def sample_two_point_crossover(indiv_a, indiv_b):
    """
    2-point crossover

    Applies 2-point crossover to the 2 parents received. The two parents are binary lists
    of the sampling component of the individuals.
    Parameters:
    ----------
    indiv_a : list
        Binary string of the sampling component of the parent A

    indiv_b : list
        Binary string of the sampling component of the parent B

    Returns:
    -------
    list, list
        Changed binary strings of the offsprings

    """
    sampling_size = len(indiv_a)
    point1 = np.random.randint(0, sampling_size)
    point2 = np.random.randint(0, sampling_size)

    # to have different crossover points
    while point1 == point2:
        point2 = np.random.randint(0, sampling_size - 1)

    if point1 > point2:
        point1, point2 = point2, point1

    offspring_a = indiv_a[:point1] + indiv_b[point1:point2] + indiv_a[point2:]
    offspring_b = indiv_b[:point1] + indiv_a[point1:point2] + indiv_b[point2:]
    return offspring_a, offspring_b
