datasets = ['Australian', 'cnae-9', 'credit-g', 'mfeat-factors']
# datasets = ['adult', 'credit-card', 'portuguese-bank-marketing']
# frameworks = {
#     'Baseline' : '../../../../../Volumes/JoSandisk/research/experiments/fairness-aware-edca/cv/edca-mcc-recall-baseline',
#     'Fairness-Aware' : '../../../../../Volumes/JoSandisk/research/experiments/fairness-aware-edca/cv/edca-fairaware-mcc-recall'
# }



# def rgb255(rgb):
#     return tuple(c / 255 for c in rgb)

# frameworks_palette = {
#     'Baseline' : rgb255((60, 60, 60)),
#     'Fairness-Aware' : rgb255((175, 175, 175))
# }

frameworks = {
    'Baseline' :  '../../../../../data/edca/experiments/30-min/paper-epm/baseline',
    'Static EPM' :  '../../../../../data/edca/experiments/30-min/paper-epm/static-epm',
    'Static EPM 50' :  '../../../../../data/edca/experiments/30-min/paper-epm/static-epm-50',
    'Dynamic EPM' :  '../../../../../data/edca/experiments/30-min/paper-epm/dynamic-epm',
    'Dynamic EPM 50' :  '../../../../../data/edca/experiments/30-min/paper-epm/dynamic-epm-50',
}

save_path = '../../../../../data/edca/experiments/30-min/paper-epm/data'
# save_path = '../tests/experiments/paper-fairness'

experimentation_name = 'estimation-paper'
# experimentation_name = 'fairness-paper'

images_dir = '../images/estimation'
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