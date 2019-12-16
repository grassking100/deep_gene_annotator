import os
import pandas as pd
import json
import pickle

def add_exist_trial(study,folder_path):
    path = os.path.join(folder_path,'trial.pkl')
    with open(path,'rb') as fp:
        trial=pickle.load(fp)
    try:
        study._storage.create_new_trial(study.study_id, template_trial=trial)
    except:
        print("Fail to load trial {}  from {}".format(trial.trial_id,path))
        raise

def add_exist_trials(study,folder_path):
    df = pd.read_csv(folder_path,sep='\t',index_col=False)
    df = df.sort_values(by=['trial_id'])['path']
    for path in list(df):
        add_exist_trial(study,path)
