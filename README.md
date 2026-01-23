# EDCA – An Evolutionary Data-Centric AutoML Framework for Efficient Pipelines

EDCA is a low-cost AutoML framework capable of creating simpler but efficient ML solutions.

## What is EDCA?

EDCA is a Python library for Automated Machine Learning (AutoML). It optimizes the entire ML pipeline. Given a classification dataset, EDCA starts by making an analysis of the features types and characteristics. This analysis serves to define the data transformations required for the data in question. Then, with the pipeline steps required, it starts the search for its bests estimators and models for each step of the pipeline. The search relies on a Genetic Algorithm. In the end, the user receives the best pipeline found ready to make predictions over unseen data.

![image info](docs/images/edca/edca-process-overview.png)

## Installation

### Conda

Install the conda environment from the yml file with all the dependencies.

    conda env create -f environment.yml

    conda activate edca

### pip

Install from the pip requirements (not recommended)

    conda create --name edca python=3.11.13

    conda activate edca 

    pip install -r requirements.txt

## Getting Started

Check the tutorial for a in-depth description on how to use EDCA.

Start by importing it

```python

from edca.evodata import DataCentricAutoML

```

An the, start the optimization using EDCA

```python

automl = DataCentricAutoML(
    task='classification', # detail the ML task
    metric='f1', # specify the search metric
    use_sampling=True, # use sampling to speed up the search
    use_feature_selection=True # use feature selection to speed up the search and improve the model generalization
)

# * optimize the ML pipeline with EDCA*
automl.fit(X_train, y_train)
```

To run the tutorial or the benchmark using Docker, change the code and configuration files, accordingly, and then run:

- for the benchmark

      docker compose -f docker-compose.benchmark.yml -p edca-benchmark up

- for the tutorial

      docker compose -f docker-compose.tutorial.yml -p edca-tutorial up

## Repository Structure

- `*EDCA/analysis*`: scripts for making a statistical analysis of the benchmarks
- `*EDCA/benchmarks*`: source code for making the benchmarks
  
  - `*EDCA/benchmarks/configs*`: configuration files to use on the benchmarks
  - `*EDCA/benchmarks/src*`: source code for the benchmarks

- `*EDCA/data*`: contains some datasets used on the benchmarks and on the tutorial.
  
  - `*EDCA/data/datasets*`: contains the datasets
  - `*EDCA/data/metadata*`: contains metadata about the OpenML datasets used

- `*EDCA/docs*`: contains the original EDCA paper detailing its process and additional documentation.
- `*EDCA/edca*`: contains EDCA implementation
- `*EDCA/tutorials*`: contains a brief hand-on tutorial to use EDCA.

Note: Inside most directories there is a *README.md* detailing its content.

## Contact

- Joana Simões (<joanasimoes@dei.uc.pt>) (corresponding author)
- João Correia (<jncor@dei.uc.pt>)

## Cite Us

```bibtex

@mastersthesis{simoes2024data,
  title={Data Centric Optimisation in AutoML},
  author={Sim{\~o}es, Joana Maria Silva},
  url={https://estudogeral.uc.pt/handle/10316/118054?locale=en},
  year={2024}
}

@inproceedings{simoes2025edca,
  title={EDCA--An Evolutionary Data-Centric AutoML Framework for Efficient Pipelines},
  author={Sim{\~o}es, Joana and Correia, Jo{\~a}o},
  booktitle={International Conference on the Applications of Evolutionary Computation (Part of EvoStar)},
  pages={71--88},
  year={2025},
  organization={Springer}
}
```
