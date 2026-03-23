import os

frameworks_palette = {'EDCA': (0.3, 0.3, 0.3)}  # grey

frameworks = {
    'edca': 'edca'
}

#start_name = 'edca'


save_path = os.path.join('..', 'results/MedViT2-nopt/exp3_mlp')
if not os.path.exists(save_path):
    os.makedirs(save_path)
# save_path = '../tests/experiments/paper-fairness'

experimentation_name = 'exp3_mlp'

LOGS_ROOT = os.path.join('..', 'logs', 'MedViT2-nopt')

datasets = list(sorted(['img_raw_tab','img_proj_tab', 'img_raw_features','3planes_raw_features' ]))


images_dir = '../images/exp1-first'
# images_dir = '../images/fairness-v2'

fairness_parameters =  {
    "adult.csv" : {
        "sensitive_attributes" : ["age", "race", "sex"],
        "positive_class" : ">50K",
        "bin_class": {
            "age" : [25, 60]
        }
    },
    "portuguese-bank-marketing.csv" : {
        "sensitive_attributes" : ["age", "marital"],
        "positive_class" : "yes",
        "bin_class": {
            "age" : [25, 60]
        }
    },
    "credit-card.csv" : {
        "sensitive_attributes" : ["x2", "x4", "x3"],
        "positive_class" : 1,
        "bin_class" : None
    },
    "diabetes-hospital.csv" : {
        "sensitive_attributes" : ["gender"],
        "positive_class" : 1,
        "bin_class": None
    }
    }