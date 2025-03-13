# EDCA's source code

The files inside this directory provide the source code to [run EDCA](#how-to-run-edca-alone), and to run the experiments to [compare EDCA with FLAML and TPOT](#how-to-run-the-experiments).

## How to run the experiments?

To run the experimentation to compare EDCA, FLAML and TPOT, use:

    python main.py {configuration file}

### Configuration file

To compare the experiments, there are several parameters that should be configured. The configuration file should have the following parameters:

<table>
    <thead>
        <tr> 
            <th>Parameter</th>
            <th>Description</th>
            <th>Value type</th>
            <th>Default Value</th>
        </tr>
    </thead>
    <tbody>
        <tr> 
            <th><code>openml_splits</code></th>
            <th>Use OpenML Splits from AMLB Benchmark or not</th>
            <th>boolean</th>
            <th><code>false</code></th>
        </tr>
        <tr> 
            <th><code>dataset</code></th>
            <th>List of the datasets to use or *null* to use the default datasets. Note that the datasets should be in the datasets folder (the default one) or the path give tono the *config_variables.py* file.</th>
            <th>List of str or null</th>
            <th><code>null</code></th>
        </tr>
        <tr> 
            <th><code>task</code></th>
            <th>Type of task.</th>
            <th>str ["regression", "classification"]</th>
            <th><code>false</code></th>
        </tr>
        <tr> 
            <th><code>kfold</code></th>
            <th>Number of folds to use in Stratified K-Fold cross-validation if *openml_splits* is false</th>
            <th>int</th>
            <th><code>5</code></th>
        </tr>
        <tr> 
            <th><code>seeds</code></th>
            <th>List of seeds to tested the frameworks</th>
            <th>List of int</th>
            <th><code>[42]</code></th>
        </tr>
        <tr> 
            <th><code>time_budget</code></th>
            <th>Time budget (in seconds) to use for the search</th>
            <th>int, -1 indicates that it has no time budget, using the *n_iterations* as stop criteria</th>
            <th><code>-1</code></th>
        </tr>
        <tr> 
            <th><code>n_iterations</code></th>
            <th>Number of iterations of the optimization algorithms. Used if *time_budget* is null</th>
            <th>int</th>
            <th><code>150</code></th>
        </tr>
        <tr> 
            <th><code>metric</code></th>
            <th>Name of the search metric. It should be proper to the type of task</th>
            <th>str</th>
            <th><code>mcc</code></th>
        </tr>
        <tr> 
            <th><code>n_jobs</code></th>
            <th>Number of workers to use in parallel search. One is equal to sequential search without threads</th>
            <th>int</th>
            <th><code>1</code></th>
        </tr>
        <tr> 
            <th><code>save_path</code></th>
            <th>Path where to store the experiments</th>
            <th>str</th>
            <th><code>""</code></th>
        </tr>
        <tr> 
            <th><code>run_fold</code></th>
            <th>Specific fold to run from the cross-validation</th>
            <th>int or null to run all the folds</th>
            <th><code>null</code></th>
        </tr>
        <tr> 
            <th><code>train_evo</code></th>
            <th>Boolean to tell if EDCA should be tested</th>
            <th>boolean</th>
            <th><code>true</code></th>
        </tr>
        <tr> 
            <th><code>train_flaml</code></th>
            <th>Boolean to tell if FLAML should be tested</th>
            <th>boolean</th>
            <th><code>true</code></th>
        </tr>
        <tr> 
            <th><code>train_tpot</code></th>
            <th>Boolean to tell if TPOT should be tested</th>
            <th>boolean</th>
            <th><code>true</code></th>
        </tr>
        <tr>
            <td colspan="4" align="center"><strong>EDCA Parameters</strong></td>
        </tr>
        <tr> 
            <th><code>verbose</code></th>
            <th>Indicates the level of log to use</th>
            <th>int</th>
            <th><code>false</code></th>
        </tr>
        <tr> 
            <th><code>search_space_config</code></th>
            <th>Search space to use.</th>
            <th>If dict, it should the contain the search space according to the task type used. If str, it should give the name of the config file, inside "edca/configs". If null, it uses the default configs</th>
            <th><code>null</code></th>
        </tr>
        <tr> 
            <th><code>population</code></th>
            <th>Population size for the EA</th>
            <th>int</th>
            <th><code>25</code></th>
        </tr>
        <tr> 
            <th><code>prob_mutation</code></th>
            <th>Probability of mutation</th>
            <th>float</th>
            <th><code>0.3</code></th>
        </tr>
        <tr> 
            <th><code>prob_crossover</code></th>
            <th>Probability of crossover</th>
            <th>float</th>
            <th><code>0.8</code></th>
        </tr>
        <tr> 
            <th><code>tournament_size</code></th>
            <th>Tournament size for selecting the parent population</th>
            <th>int</th>
            <th><code>3</code></th>
        </tr>
        <tr> 
            <th><code>elitism_size</code></th>
            <th>Elitism size (to save for the next generation, even after population's restarts)</th>
            <th>int</th>
            <th><code>1</code></th>
        </tr>
        <tr> 
            <th><code>alpha</code></th>
            <th>Weight to give to the search metric component (between 0 and 1) of the fitness functions.</th>
            <th>float</th>
            <th><code>1</code></th>
        </tr>
        <tr> 
            <th><code>beta</code></th>
            <th>Weight to give to the percentage of data used (between 0 and 1) on the fitness function</th>
            <th>float</th>
            <th><code>0</code></th>
        </tr>
        <tr> 
            <th><code>gama</code></th>
            <th>Weight to give to the CPU Time component (between 0 and 1) on the fitness function</th>
            <th>float</th>
            <th><code>0</code></th>
        </tr>
        <tr> 
            <th><code>delta</code></th>
            <th>Weight to give to the data balance metric (between 0 and 1) on the fitness function</th>
            <th>int</th>
            <th><code>0</code></th>
        </tr>
        <tr> 
            <th><code>sampling</code></th>
            <th>Whither or not to use instance selection</th>
            <th>boolean</th>
            <th><code>true</code></th>
        </tr>
        <tr> 
            <th><code>feature_selection</code></th>
            <th>Whither or not to use feature selection</th>
            <th>boolean</th>
            <th><code>true</code></th>
        </tr>
        <tr> 
            <th><code>automatic_data_optimization</code></th>
            <th>Whither or not to use automatic data optimization for the instance and feature selection</th>
            <th>boolean</th>
            <th><code>true</code></th>
        </tr>
        <tr> 
            <th><code>binary_sampling_component</code></th>
            <th>Whither or not to use a binary representation for instance and feature selection genes. Not that in large datasets, it may increase the computational costs.</th>
            <th>boolean. If false, it uses a integer sparse representation.</th>
            <th><code>false</code></th>
        </tr>
        <tr> 
            <th><code>class_balance_mutation</code></th>
            <th>Whither or not to use the probability associated to each class to select the instances. Note that it only works on classification tasks that use a binary representation of the selection genes (*binary_sampling_component=True*).</th>
            <th>boolean</th>
            <th><code>0</code></th>
        </tr>
        <tr> 
            <th><code>mutation_factor</code></th>
            <th>Mutation factor to add to the probabilities when balancing the classes (*class_balance_mutation=True*)</th>
            <th>float</th>
            <th><code>0.5</code></th>
        </tr>
        <tr> 
            <th><code>uniform_crossover</code></th>
            <th>Whither or not to use a uniform crossover. If false, it uses a X point crossover</th>
            <th>boolean</th>
            <th><code>false</code></th>
        </tr>
        <tr> 
            <th><code>patience</code></th>
            <th>Number of generations without improvement to wait until restarting the population.</th>
            <th>int or null (no restarts)</th>
            <th><code>null</code></th>
        </tr>
        <tr> 
            <th><code>early_stop</code></th>
            <th>Number of generations to wait until finish the optimization process without improvement. If null, it will not finish</th>
            <th>int or null</th>
            <th><code>null</code></th>
        </tr>
        <tr> 
            <th><code>validation_size</code></th>
            <th>Size of the internal validation used to test the individuals. It uses *validation_size* x *{# Instances}* for validation, and the remaining to train.</th>
            <th>float</th>
            <th><code>0.25</code></th>
        </tr>
        <tr> 
            <th><code>mutation_percentage_change</code></th>
            <th>Percentage of the dimension to change in the mutation (integer representation)</th>
            <th>float</th>
            <th><code>0.1</code></th>
        </tr>
        <tr> 
            <th><code>mutation_size_neighborhood</code></th>
            <th>Size of the neighborhood to change in the mutation (integer representation)</th>
            <th>int</th>
            <th><code>10</code></th>
        </tr>
        <tr> 
            <th><code>flaml_ms</code></th>
            <th>Whither or not to use FLAML for the model selection and HPO. Still a work in progress.</th>
            <th>boolean</th>
            <th><code>false</code></th>
        </tr>
        <tr>
            <td colspan="4" align="center"><strong>FLAML Parameters</strong></td>
        </tr>
        <tr> 
            <th><code>flaml_hpo_method</code></th>
            <th>Search algorithm for FLMAL</th>
            <th>str ["cfo", "blend"]</th>
            <th><code>"cfo"</code></th>
        </tr>
        <tr> 
            <th><code>early_stop</code></th>
            <th></th>
            <th>int or null</th>
            <th><code>0</code></th>
        </tr>
        <tr>
            <td colspan="4" align="center"><strong>TPOT Parameters</strong></td>
        </tr>
        <tr> 
            <th><code>tpot_template</code></th>
            <th>Template for TPOT individuals </th>
            <th>str or null</th>
            <th><code>null</code></th>
        </tr>
        <tr> 
            <th><code>population</code></th>
            <th>Population size for the EA</th>
            <th>int</th>
            <th><code>25</code></th>
        </tr>
    </tbody>
</table>

### Notes

- The regression task is still a work in progress on EDCA, so the integration to use it on FLAML and TPOT is still not possible.
- By default, the experiments use the datasets folder in **'../datasets'** from the main.py level directory and save the experiments in current directory (main.py level). To use other paths, create a file "*config_variables.py*" with the following content:

```python
SAVE_DIR = {path to save the experiments}

DATASETS_SRC_DIR = {path to the datasets}
```

## How to run EDCA alone?
Notebook  provides an example on how to use EDCA for a dataset.
