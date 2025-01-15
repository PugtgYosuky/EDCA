"""Auxiliary functions for statistical analysis"""

from scipy import stats
import pandas as pd
import numpy as np


def test_normal_ks(data):
    """Kolgomorov-Smirnov"""
    norm_data = (data - np.mean(data))/(np.std(data)/np.sqrt(len(data)))
    return stats.kstest(norm_data,'norm')


def levene(group_a, group_b):
    """Test of equal variance"""
    return stats.levene(group_a, group_b)


def statistical_test_repeated(group_a, group_b, significance_level=0.05):
    " Statistical pipeline when we have the same initial conditions"
    # print('- Two categories\n- same initial conditions\n\n')

    # testes de normalidade
    norm_a = test_normal_ks(group_a)
    norm_b = test_normal_ks(group_b)
    # teste de variancia
    variance = levene(group_a, group_b)

    # print('Normality H0: The sample follows a normal distribution')
    # print('Normality Ha: The sample does not follow a normal distribution')

    # print('Variance H0: The variance is equal ')
    # print('Variance Ha: The variance is different')
    
    # print('Normality test: grupo a\n', norm_a)
    # print('\nNormality test: grupo b\n', norm_b)
    # print('\nLevene test: grupo vaginal\n', variance)

    parametric = norm_a.pvalue >= significance_level and \
        norm_b.pvalue >= significance_level and \
        variance.pvalue >= significance_level
    
    # print('Parametric test: ', parametric)


    # escolha do teste estatistico
    
    if parametric:
        # independent t-test
        stat_test = stats.ttest_rel(group_a, group_b)
    else:
        # mann- Whitney
        stat_test = stats.wilcoxon(group_a, group_b)
    # print('\n\nStatistical test:\n',stat_test)
    return stat_test


def statistical_test_independent(group_a, group_b, significance_level=0.01):
    # " Statistical pipeline when we have different initial conditions"
    # print('- Two categories\n- different initial conditions\n\n')

    # testes de normalidade
    norm_a = test_normal_ks(group_a)
    norm_b = test_normal_ks(group_b)
    # teste de variancia
    variance = levene(group_a, group_b)

    # print('Normality H0: The sample follows a normal distribution')
    # print('Normality Ha: The sample does not follow a normal distribution')

    # print('Variance H0: The variance is equal ')
    # print('Variance Ha: The variance is different')
    
    # print('Normality test: grupo a\n', norm_a)
    # print('\nNormality test: grupo b\n', norm_b)
    # print('\nLevene test: grupo vaginal\n', variance)

    parametric = norm_a.pvalue >= significance_level and \
        norm_b.pvalue >= significance_level and \
        variance.pvalue >= significance_level
    
    # print('Parametric test: ', parametric)


    # escolha do teste estatistico
    
    if parametric:
        # independent t-test
        stat_test = stats.ttest_ind(group_a, group_b)
    else:
        # mann- Whitney
        stat_test = stats.mannwhitneyu(group_a, group_b)
    # print('Statistical test:\n',stat_test)
    return stat_test

