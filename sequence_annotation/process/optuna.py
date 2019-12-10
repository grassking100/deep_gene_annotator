import os
import pandas as pd
import json
import pickle

def add_exist_trial(study,folder_path):
    with open(os.path.join(folder_path,'trial.pkl'),'rb') as fp:
        trial=pickle.load(fp)
    study._storage.create_new_trial(study.study_id, template_trial=trial)

def add_exist_trials(study,folder_path):
    df = pd.read_csv(folder_path,sep='\t',index_col=False)
    df = df.sort_values(by=['trial_id'])['path']
    for path in list(df):
        add_exist_trial(study,path)
