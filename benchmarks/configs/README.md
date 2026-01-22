# 📖 Experimentation Configuration Guide

The benchmarks use a configuration *.json* file to control and reproduce experiments. This document describes the required and optional configuration fields, their purpose, and how to modify them for different experimental runs. The configuration to control EDCA is defined on its source directory (*/EDCA/edca/README.md*).

The benchmark uses a series of evaluations to run EDCA alone, or against two state-of-the-art AutoML frameworks (FLAML and TPOT).

The *.json* file centralizes all experiment parameters, including:

- Experiments metadata
- Data sources and preprocessing
- EDCA settings
- Training and evaluation parameters
- Logging and output options
- Reproducibility controls

## CONFIGURATION PARAMETERS

`openml_splits` [bool]:  
Defines if it should use the OpenML splits for the cross-validation or not.
**Default:** `False`

`dataset` [[str] or list[str]]:  
A list with the datasets to use on the experiments. The name of the datasets, as the dataset source should be given in *benchmarks/src/config_variables.py*
**Default:** mandatory

`save_path` [str]:  
Defines the path where to store the experiments logs.
**Default:** `""`

`kfold` [int]:  
Defines the number of folds to divide the data into in cross-validation. In *main.py* file
**Default:** `5`

`run-fold` [int]:  
If given and not None, it details which fold only should be run. In *main.py* file
**Default:** `None`

`test_size` [float]:  
Defines the percentage of data that should be saved for testing. In *main_train_test.py* file
**Default:** `0.3`

`run_all_seeds` [bool]:  
It details if it should run all random seeds or only the one defined by `seed_pos`
**Default:** `True`

`seed_pos` [int]:  
It details the start of the random seeds list to run
**Default:** `0`

`train_edca` [bool]:  
Whether or not to train and benchmark EDCA
**Default:** `True`

`train_flaml` [bool]:  
Whether or not to train and benchmark FLAML
**Default:** `False`

`train_tpot` [bool]:  
Whether or not to train and benchmark TPOT
**Default:** `False`

`flaml_hpo_method` [str]:  
Details the FLAML search algorithm. See more in [FLAML website](https://microsoft.github.io/FLAML/docs/reference/automl/automl)
**Default:** `cfo`

`tpot_template` [str]:  
Details the structure for TPOT solutions [TPOT website](https://epistasislab.github.io/tpot/latest/tpot_api/classifier/)
**Default:** `cfo`

`fairness_parameters` [dict]:  
A dictionary detailing data characteristics for when using fairness-aware optimization. It should contains an entry per dataset, and the values should contains the *sensitive_attributes*, metrics weights, *positive_class*, and *bin_class* detailing all numeric features that should be converted to bins. e.g.:

```python
    {
        "sensitive_attributes" : ["age", "race", "sex"],
        "demographic_parity": 0,
        "equal_opportunity": 0,
        "equalized_odds": 0,
        "abroca" : 0,
        "positive_class" : ">50K",
        "bin_class": {
            "age" : [25, 60]
        }
    }
```

**Default:** `cfo`
