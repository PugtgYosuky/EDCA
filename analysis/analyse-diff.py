# %%
import json
from parameters import datasets, frameworks
import os

# %%
original_seeds = [42, 384, 518, 522, 396, 400, 23, 791, 666, 283, 28, 298, 557, 309, 822, 569, 825, 185, 574, 325, 844, 90, 219, 864, 872, 618, 747, 365, 237, 767]
# %%
for dataset in datasets:
    for framework_name, path in frameworks.items():
        seeds = {}
        experiments = os.listdir(os.path.join(path, dataset))
        experiments = [exp for exp in experiments if exp.startswith('exp')]
        for exp in sorted(experiments):
            with open(os.path.join(path, dataset, exp, 'config.json')) as file:
                config = json.load(file)
                seeds[config['seed']] = seeds.get(config['seed'], []) + [exp]

        print(framework_name, '-', dataset, '-', len(experiments), 'exps', '-', len(seeds.keys()), 'seeds')
        for seed, values in seeds.items():
            if len(values) > 1:
                print(values)
        print('Left:' , set(original_seeds) - set(seeds.keys()))