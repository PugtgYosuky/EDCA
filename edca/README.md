# 🚀 EDCA: Evolutionary Data-Centric AutoML Framework

Welcome to the **EDCA Hyperparameter Guide**.

Unlike traditional model-centric AutoML tools that focus primarily on model selection, EDCA is designed to replicate the workflow of a data science expert. It treats data as the primary driver of the optimization process, automatically generating end-to-end machine learning pipelines that adapt to the unique characteristics of your dataset.

Each pipeline in EDCA is represented as an "individual" in a population. By adjusting the hyperparameters in this guide, you control how these individuals evolve over time to reach the optimal balance between model accuracy and computational efficiency.

## EDCA Parameters

`verbose` [int]:  
Defines which populations should be saved during optimization.  
-1 = all, 0 = only initial & final, or a positive integer = save every N generations.  
**Default:** `-1`

`search_space_config` [dict | str | None]:  
Indicates the search space to use during the optimization. If *None*, it uses the default predefined search space for the type of tasks given. If *dict*, it uses that search space. If str, it uses the path given to load the configuration JSON with the search space.
**Default:** `None`

`population` [int]:  
Size of the population in the evolutionary algorithm (EA).  
**Default:** `25`

`prob_mutations` [float]:  
Probability of the mutation of an individual in the EA.  
**Default:** `0.3`

`prob_crossover` [float]:  
Probability of crossover in the EA.  
**Default:** `0.8`

`tournament_size` [int]:  
Size of the tournament in the EA.  
**Default:** `3`

`elitism_size` [int]:  
Number of best individuals to keep across generations.  
**Default:** `1`

`binary_sampling_component` [bool]:  
Use a binary representation for the data reduction genes or a integer one.
**Default:** `True`

`automatic_data_optimization` [bool]:  
Use automatic data optimization to select instances/features.  
**Default:** `True`

`sampling` [bool]:  
Whether or not to use instance selection.  
**Default:** `True`

`feature_selection` [bool]:  
Whether or not to use feature selection.  
**Default:** `True`

`fitness_params` [dict]:  
Details the weights associated to each component of the fitness function. See details below.
**Default:** `{"metric" : 1.0}`

`class_balance_mutation` [bool]:  
Apply balance mutation to the sampling component.  
**Default:** `false`

`mutation_factor` [float]:  
Mutation factor to use in the balance mutation of the EA.  
**Default:** `0.5`

`uniform_crossover` [bool]:  
Whether to apply uniform crossover (`true`) or point crossover (`false`).  
**Default:** `true`

`patience` [int | None]:  
Generations to wait without improvement before restarting population.  
**Default:** `null`

`early_stop` [int | None]:  
Generations to wait without improvement before finishing optimization.  
**Default:** `null`

`sampling_start` [int]:  
Generation number after which to stop data selection.  
**Default:** `0`

`mutation_size_neighborhood` [int]:  
Size of the surrounding search space for each operator mutation.  
**Default:** `10`

`mutation_percentage_change` [float]:  
Percentage of change in each mutation.  
**Default:** `0.1`

`flaml_ms` [bool]:  
Whether to use FLAML for model selection and HPO (experimental).  
**Default:** `False`

`fairness_params` [dict]:  
Dictionary with parameters for fairness-aware optimization. See details below.
**Default:** `{}`

`optimize_preprocessing`[bool]:
Whether or not to include and optimize data preprocessing based on data characteristics

**Default:** `True`

### Fitness component parameters

The fitness functions can contain several components. Therefore, each component has a weight associated so that in the end we could calculate the weighted sum to calculate the fitness of the individuals. These components evaluate several aspects of the individuals. The parameters should be

  `metric` [float]:  
  Weight associated with the optimization metrics defined by the user. It calculates the performance of the predictions.
  **Default:** `1.0`

  `data_size` [float]:  
  Weight associated with the size of the dataset selected by the individual.
  **Default:** `0.0`

  `training_time` [float]:  
  Weight associated with time used to evaluated the individual.
  **Default:** `0.0`

  `balance_metric` [float]:  
  Weight associated with balance of the classes of the selected dataset.
  **Default:** `0.0`

  `fairness_metric` [float]:  
  Weight associated with fairness metrics. There are three metrics that can be used. Note that the *fairness_params* should be given to use a fairness-aware system. See section below for more details.
  **Default:** `0.0`
  
### Fairness-aware optimization

To use the fairness-aware optimization, you need to give more information to EDCA using the `fairness_params` parameter. This entry should contain a `dict` detailing the fairness attributes.

The information should be:

  `sentitive_attributes` [list[str]]: Sensitive features to protect. **Default:** : `[]`

  `demographic_parity` [float]: Weight to give to the metric on its part of the fitness function. **Default:** `0`

  `equal_opportunity` [float]: Weight to give to the metric on its part of the fitness function. **Default:** `0`

  `equalized_odds` [float]: Weight to give to the metric on its part of the fitness function. **Default:** `0`

  `positive_class` [integer, float, str]: Says which is the positive class of the target. **Default:** `null`

  `bin_class` [dict]: Indicates which numerical sensitive attributes should be encoded (e.g. age). Each entry of dict should contain the name of the attribute as key, and the values are a list with the cut points for the ranges. E.g. age with values [25, 60] will transform the numerical feature into the categories: <25, 25-60, >60. **Default:** `null`
