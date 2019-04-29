import pandas as pd
from matplotlib_venn import venn2

def compare(path_1,path_2,name_1,name_2,header="infer"):
    lhs = pd.read_csv(path_1,sep='\t',header=header)
    rhs = pd.read_csv(path_2,sep='\t',header=header)
    lhs = set(lhs[name_1])
    rhs = set(rhs[name_2])
    lhs = {x for x in lhs if x==x}
    rhs = {x for x in rhs if x==x}
    venn2([lhs,rhs])

def diff(path_1,path_2,name_1,name_2,reverse=False,header="infer"):
    lhs = pd.read_csv(path_1,sep='\t',header=header)
    rhs = pd.read_csv(path_2,sep='\t',header=header)
    lhs = set(lhs[name_1])
    rhs = set(rhs[name_2])
    lhs = {x for x in lhs if x==x}
    rhs = {x for x in rhs if x==x}
    if not reverse:
        return lhs - rhs
    else:
        return rhs - lhs