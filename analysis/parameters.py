datasets = ['exp2_MedViT-nopt']
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
frameworks_palette = {'EDCA': (0.3, 0.3, 0.3)}  # grey

frameworks = {
    'EDCA': '../logs/exp1/testing/exp2_MedViT-nopt/exp_2026-01-29 15:41:04.605647'
}

start_name = 'edca'

save_path = '../logs/exp1/testing/exp2_MedViT-nopt/exp_2026-01-29 15:41:04.605647'
# save_path = '../tests/experiments/paper-fairness'

experimentation_name = 'exp1-first'
# experimentation_name = 'fairness-paper'

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