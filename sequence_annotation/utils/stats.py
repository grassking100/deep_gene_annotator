import numpy as np
import pandas as pd
from scipy.stats import median_absolute_deviation
from rpy2 import robjects
from rpy2.robjects.packages import importr


def mad_threshold(values, coef=None):
    coef = coef or 1
    mad = median_absolute_deviation(values)
    threshold = np.median(values) + mad * coef
    return threshold


def _conver_r_list(list_):
    return 'c(' + ','.join([str(v) for v in list_]) + ')'


def exact_wilcox_rank_sum_test(lhs, rhs, alternative):
    """The method to calculate Wilcoxon Rank Sum Test by R's wilcox.exact function"""
    importr('exactRankTests')
    COMMAND = "wilcox.exact({},{},paired=F,exact=T,alternative='{}')$p.value"
    lhs = _conver_r_list(lhs)
    rhs = _conver_r_list(rhs)
    command = COMMAND.format(lhs, rhs, alternative)
    p_value = np.array(robjects.r(command))[0]
    return p_value

def exact_wilcox_rank_sum_compare(lhs_df, rhs_df, lhs_name, rhs_name, targets=None, threshold=None):
    """The method to calculate p values of dataframe's data by exact_wilcox_rank_sum_test"""
    threshold = threshold or 0.05
    lhs_median_name = 'median({})'.format(lhs_name)
    rhs_median_name = 'median({})'.format(rhs_name)
    if set(lhs_df.columns) != set(rhs_df.columns):
        raise Exception("The columns of lhs_df and rhs_df are not the same")
    targets = set(lhs_df.columns)    
    lhs_dict = lhs_df.fillna(0).to_dict('list')
    rhs_dict = rhs_df.fillna(0).to_dict('list')
    table = []
    for name in lhs_df.columns:
        if name in targets:
            lhs_values = lhs_dict[name]
            rhs_values = rhs_dict[name]
            lhs_median = np.median(lhs_values)
            rhs_median = np.median(rhs_values)

            if lhs_median < rhs_median:
                compare = 'greater'
            elif lhs_median == rhs_median:
                compare = 'two.sided'
            else:
                compare = 'less'

            p_val = exact_wilcox_rank_sum_test(rhs_values, lhs_values, compare)
            item = {}
            item['metric'] = name
            item[lhs_median_name] = lhs_median
            item[rhs_median_name] = rhs_median
            item['compare'] = compare
            item['p_val'] = p_val
            item['threshold'] = threshold
            item['pass'] = p_val <= threshold
            item['test'] = 'Wilcoxon Rank Sum Test'
            table.append(item)

    columns = [lhs_median_name, rhs_median_name, 'metric',
               'compare', 'p_val', 'threshold', 'pass', 'test']
    return pd.DataFrame.from_dict(table)[columns]
