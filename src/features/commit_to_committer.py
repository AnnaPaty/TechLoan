import numpy as np
import pandas as pd
import time

# Find the data
import os
datapath = os.getcwd().split("TechLoan")[0]+"TechLoan/data/interim/"


print("Starting aggregation commit to committer... ", end="")
start = time.time()

df = pd.read_csv(datapath+"commit-level_dataframe.csv")
df_project = df.pivot_table(values=['PROJECT_ID'],
                            index=['COMMITTER'],
                            aggfunc={'PROJECT_ID':lambda x: len(x.unique())})
df_commit_hash = df.pivot_table(values=['COMMIT_HASH'],
                            index=['COMMITTER'],
                            aggfunc='count')
df_mean = df.pivot_table(values=['AVG_LINES_ADDED','AVG_LINES_REMOVED','COUNT_SOLVED','AVG_DURATION','COUNT_REFACTS','FIXED_ISSUES','INDUCED_ISSUES','FILES_CHANGED'],
                            index=['COMMITTER'],
                            aggfunc='mean')
df_committer = pd.merge(pd.merge(df_mean,df_commit_hash,on='COMMITTER'),df_project,on='COMMITTER')

df_committer.reset_index(level=0, inplace=True)
df_committer.to_csv("committer-level_dataframe.csv", index=False)

print(f"ended successfully ({round(time.time()-start, 2)} sec)")
