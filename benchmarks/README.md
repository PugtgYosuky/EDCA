# 📊 Benchmark code to compare EDCA with other AutoML frameworks

The directory contains the source code to evaluate and compare EDCA with other frameworks.

## 🚀 Run the benchmark

Use the `main.py` file and a configuration `.json` to run a set of experimental runs. The configuration file should contain the parameters defined in `configs/README.md`. The `main.py` script uses cross-validation to evaluate EDCA on the given tabular dataset.

To run the experiments under a predefined configuration file, use

    python main.py {configuration file .json}

If you have a set of configuration .json files, and want to run them all sequentially, use:

    python main.py {path to configurations folder}

❗️Note: if you want to use a train-test evaluation instead of a cross-validation evaluation, use `main_train_test.py` instead of `main.py`

In the end, this stores all the information, which could be analyzed using the files under *benchmarks/analysis*.
