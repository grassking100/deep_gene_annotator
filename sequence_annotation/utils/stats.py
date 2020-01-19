import numpy as np
from scipy.stats import median_absolute_deviation
from rpy2 import robjects
from rpy2.robjects.packages import importr

def mad_threshold(values,coef=None):
    coef=  coef or 1
    mad = median_absolute_deviation(values)
    threshold =  np.median(values) + mad*coef
    return threshold

def _conver_r_list(list_):
    return 'c('+','.join([str(v) for v in list_])+')'

def exact_wilcox_signed_rank_test(lhs,rhs,alternative):
    """The method to calculate Wilcoxon Signed Rank Test by R's wilcox.exact function"""
    importr('exactRankTests')
    COMMAND = "wilcox.exact({},{},paired=T,exact=T,alternative='{}')$p.value"
    command = COMMAND.format(_conver_r_list(lhs),_conver_r_list(rhs),alternative)
    p_value=np.array(robjects.r(command))[0]
    return p_value
    
def exact_wilcox_signed_rank_compare(df,lhs_name,rhs_name,threshold=None):
    """The method to calculate p values of dataframe's data by exact_wilcox_signed_rank_test"""
    COMMAND = "wilcox.test({},{},paired=T,exact=T,correct=F,alternative='{}')$p.value"
    threshold = threshold or 0.05
    table = []
    targets = set(df['target'])
    lhs_median_name = 'median({})'.format(lhs_name)
    rhs_median_name = 'median({})'.format(rhs_name)
    for target in targets:
        lhs_df = df[(df['target']==target) & (df['name']==lhs_name)].sort_values('source')
        rhs_df = df[(df['target']==target) & (df['name']==rhs_name)].sort_values('source')
        lhs_values = list(lhs_df['value'])
        rhs_values = list(rhs_df['value'])
        lhs_median = np.median(lhs_values)
        rhs_median = np.median(rhs_values)

        if lhs_median < rhs_median:
            status ='less'
        elif lhs_median == rhs_median:
            status = 'two.sided'
        else:
            status ='greater'
        
        p_val = exact_wilcox_signed_rank_test(lhs_values,rhs_values,status)
        item = {}
        item['target'] = target
        item[lhs_median_name] = lhs_median
        item[rhs_median_name] = rhs_median
        item['status'] = status
        item['p_val'] = p_val
        item['pass'] = p_val<=threshold
        table.append(item)

    columns = ['target','status','p_val','pass',lhs_median_name,rhs_median_name]
    return pd.DataFrame.from_dict(table)[columns]
