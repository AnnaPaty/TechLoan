import numpy as np
import pandas as pd
import time

# Find the data
import os
datapath = os.getcwd().split("TechLoan")[0]+"TechLoan/data/raw/"

pipeline_start = time.time()

## Read tables from drive ##
print("Reading tables... ", end="")
start = time.time()

# Load git_commits_changes table by chunks to fit RAM
data, colnames = [], ["COMMIT_HASH","LINES_ADDED","LINES_REMOVED"]
reader = pd.read_csv(datapath + "GIT_COMMITS_CHANGES.csv", chunksize=1e4, lineterminator='\n',
                                  usecols = colnames)
for chunk in reader:
    for hash,added,removed in zip(chunk["COMMIT_HASH"], chunk["LINES_ADDED"], chunk["LINES_REMOVED"]):
        data.append([hash,added,removed])
git_commits_changes = pd.DataFrame(data, columns=colnames)

git_commits = pd.read_csv(datapath + "GIT_COMMITS.csv", usecols=["COMMIT_HASH", "PROJECT_ID", "COMMITTER"])

jira_issues = pd.read_csv(datapath + "JIRA_ISSUES.csv", usecols=["HASH","CREATION_DATE","RESOLUTION_DATE"])

refactoring_miner = pd.read_csv(datapath + "REFACTORING_MINER.csv", usecols=["COMMIT_HASH", "REFACTORING_TYPE"])

szz_fault_inducing_commits = pd.read_csv(datapath + "SZZ_FAULT_INDUCING_COMMITS.csv", usecols=["FAULT_FIXING_COMMIT_HASH", "FAULT_INDUCING_COMMIT_HASH"])

print(f"done ({round(time.time()-start, 2)}s)")

## Checking quality ##
print("Checking data quality... ")

# Quality of git_commits: COMMITTER
print("\n[Quality check] Git commits")
perc_noauth = round(len(git_commits[git_commits["COMMITTER"]=="No Author"])/len(git_commits)*100, 4)
print(f"Percentage of rows with 'No Author' as committer: {perc_noauth}%")

is_NaN = git_commits.isnull()
row_has_NaN = is_NaN.any(axis=1)
perc_nan = round(sum(row_has_NaN)/len(git_commits)*100, 4)
print(f"Percentage of rows with NaN as committer: {perc_nan}%")


# Quality of git_commits: PROJECT_ID
for i,projid in enumerate(pd.unique(git_commits["PROJECT_ID"])):
    print(f"Project ID ({i}): {projid}")


# Quality of COMMIT_HASH in git_commits and git_commits_changes
hash_len = len('52fc76012c5f969145c39d3fb398a7c2c094474f')
type_fault = len_fault = 0
for h in np.unique(np.concatenate((np.array(git_commits["COMMIT_HASH"]), git_commits_changes["COMMIT_HASH"]))):
    if type(h) is not str:
        type_fault += 1
    elif len(h) != hash_len:
        len_fault += 1
print(f"# of type faults: {type_fault}\n# of length fault: {len_fault}")

# Quality of GIT_COMMITS_CHANGES: LINES_ADDED & LINES_REMOVED
print("\n[Quality check] Git commits changes")
print(git_commits_changes["LINES_ADDED"].describe())
print(git_commits_changes["LINES_REMOVED"].describe())
type_fault = 0
for n in git_commits_changes["LINES_ADDED"].append(git_commits_changes["LINES_REMOVED"]):
    type_fault += type(n) is not int
print(f"# of type faults: {type_fault}")

# Quality of JIRA_ISSUES: CREATION_DATE and RESOLUTION_DATE to datetime

jira_issues["CREATION_DATE"] = pd.to_datetime(jira_issues["CREATION_DATE"], utc=True)
jira_issues["RESOLUTION_DATE"] = pd.to_datetime(jira_issues["RESOLUTION_DATE"], utc=True)

# Quality of REFACTORING_MINER: removing NaNs
print("\n[Quality check] Refactoring miner")
for i,h in enumerate(refactoring_miner["COMMIT_HASH"]):
    if type(h) is not str:
        print(f"Dropping entry #{i} with COMMIT_HASH {h}, {type(h)}")
        refactoring_miner.drop(labels=i, axis=0, inplace=True)

# Quality of REFACTORING_MINER: REFACTORING_TYPE
for i,rt in enumerate(pd.unique(refactoring_miner["REFACTORING_TYPE"])):
    print(f"Refactoring type #{i}: {rt}")

# Quality of COMMIT_HASH in GIT_COMMITS, GIT_COMMITS_CHANGES, REFACTORING_MINER, JIRA_ISSUES and SZZ_FAULT_INDUCING_COMMITS7
# OBS: unique would have failed if there were any NaNs!
all_hashes = np.unique(np.concatenate((np.array(git_commits["COMMIT_HASH"]),
                                       np.array(git_commits_changes["COMMIT_HASH"]),
                                       np.array(jira_issues["HASH"]),
                                       np.array(refactoring_miner["COMMIT_HASH"]),
                                       np.array(szz_fault_inducing_commits["FAULT_FIXING_COMMIT_HASH"]),
                                       np.array(szz_fault_inducing_commits["FAULT_INDUCING_COMMIT_HASH"]))))
hash_len = len('52fc76012c5f969145c39d3fb398a7c2c094474f')
type_fault = len_fault = 0
for h in all_hashes:
    if type(h) is not str:
        type_fault += 1
    elif len(h) != hash_len:
        len_fault += 1
print(f"# of type faults: {type_fault}\n# of length fault: {len_fault}")

# Data cleaning based on the previous quality check
git_commits = git_commits[~row_has_NaN]
git_commits = git_commits[git_commits["COMMITTER"]!="No Author"]

## Feature Engineering ##
print("Feature engineering... ", end="")
start = time.time()

# Compute duration for the resolution of the issue
jira_issues["DURATION"] = [(jira_issues["RESOLUTION_DATE"][i] - jira_issues["CREATION_DATE"][i]).total_seconds() for i in range(len(jira_issues["RESOLUTION_DATE"]))]

# Date columns are no longer necessary
jira_issues.drop('RESOLUTION_DATE', inplace=True, axis=1)
jira_issues.drop('CREATION_DATE', inplace=True, axis=1)

# Remove negative durations
jira_issues = jira_issues[jira_issues["DURATION"] > 0]

# Obtain number of FIXED_ISSUES and INDUCED_ISSUES from szz_fault_inducing_commits table
szz_fault_inducing_commits['ONES'] = [1]*len(szz_fault_inducing_commits['FAULT_INDUCING_COMMIT_HASH'])
szz_fixing = pd.pivot_table(szz_fault_inducing_commits, values=['ONES'], index=['FAULT_FIXING_COMMIT_HASH'], aggfunc={'ONES': np.sum}).reset_index(level=0)
szz_fixing.rename(columns={'ONES': 'FIXED_ISSUES', 'FAULT_FIXING_COMMIT_HASH': 'COMMIT_HASH'}, inplace=True)
szz_inducing = pd.pivot_table(szz_fault_inducing_commits, values=['ONES'], index=['FAULT_INDUCING_COMMIT_HASH'], aggfunc={'ONES': np.sum}).reset_index(level=0)
szz_inducing.rename(columns={'ONES': 'INDUCED_ISSUES', 'FAULT_INDUCING_COMMIT_HASH': 'COMMIT_HASH'}, inplace=True)

print(f"done ({round(time.time()-start, 2)}s)")


## Statistical Analysis ##
print("Running statistical analysis... ")

def outliers(data, r=1.5, info=False):
    """
    Returns an array of booleans indicating which entries are considered outliers
    based on the boundary defined with the Interquartile Range (IQR) times the allowance r.
    """
    v = np.array(data)
    nbefore = len(v)
    out = np.array([False]*nbefore)
    Q1 = np.percentile(v, 25, interpolation = 'midpoint')
    Q3 = np.percentile(v, 75, interpolation = 'midpoint')
    IQR = Q3 - Q1
    out[v > Q3+r*IQR] = True
    out[v < Q1-r*IQR] = True
    if info:
        print(f"{sum(out)} outliers were detected ({round(sum(out)/nbefore*100, 2)}% of the total).")
    return out


# Outliers: DURATION, jira_issues
out = outliers(jira_issues["DURATION"], r=5, info=True)
jira_issues = jira_issues[~out]

# Outliers: LINES_ADDED + LINES_REMOVED, git_commits_changes
out_added = outliers(git_commits_changes["LINES_ADDED"], r=5, info=True)
out_removed = outliers(git_commits_changes["LINES_REMOVED"], r=5, info=True)
out = np.array([out_added[i] or out_removed[i] for i in range(len(out_added))])   # remove entry if outlier in any of the columns
git_commits_changes = git_commits_changes[~out]


## Aggregation -- Commit level ##
print("Aggregating tables... ", end="")
start = time.time()

# jira_issues: aggregate by commit --> avg duration + count issues solved per commit
jira_issues['COUNT_SOLVED'] = [1]*len(jira_issues['DURATION'])
jira_issues = pd.pivot_table(jira_issues, values=['DURATION', 'COUNT_SOLVED'], 
                             index=['HASH'], aggfunc={'DURATION': np.mean, 'COUNT_SOLVED': np.sum})
jira_issues.rename(columns={'HASH': 'COMMIT_HASH', 'DURATION': 'AVG_DURATION'}, inplace=True)

# git_commits_changes: aggregate by commit --> number of modified files + avg number of lines added and deleted
git_commits_changes['FILES_CHANGED'] = [1]*len(git_commits_changes['COMMIT_HASH'])
git_commits_changes = pd.pivot_table(git_commits_changes, values=['LINES_ADDED', 'LINES_REMOVED', 'FILES_CHANGED'],
                                    index=['COMMIT_HASH'],
                                    aggfunc={'LINES_ADDED': np.mean, 'LINES_REMOVED': np.mean, 'FILES_CHANGED': np.sum})
git_commits_changes.rename(columns={'LINES_ADDED': 'AVG_LINES_ADDED', 'LINES_REMOVED': 'AVG_LINES_REMOVED'}, inplace=True)

refactoring_miner['COUNT_REFACTS'] = [1]*len(refactoring_miner['COMMIT_HASH'])
refactoring_miner = pd.pivot_table(refactoring_miner, values=['COUNT_REFACTS'],
                                   index=['COMMIT_HASH'], aggfunc={'COUNT_REFACTS': np.sum})

print(f"done ({round(time.time()-start, 2)}s)")


## Integrate tables using COMMIT_HASH ##
print("Joining tables... ", end="")
start = time.time()
df_comm = git_commits.merge(git_commits_changes, on="COMMIT_HASH", how="inner")
df_comm = df_comm.join(jira_issues, on="COMMIT_HASH")
df_comm['COUNT_SOLVED'].replace(np.nan, 0, inplace=True)
df_comm['AVG_DURATION'].replace(np.nan, 0, inplace=True)
df_comm = df_comm.join(refactoring_miner, on="COMMIT_HASH")
df_comm['COUNT_REFACTS'].replace(np.nan, 0, inplace=True)
df_comm.set_index("COMMIT_HASH", inplace=True)
szz_fixing.set_index("COMMIT_HASH", inplace=True)
df_comm = df_comm.join(szz_fixing, on="COMMIT_HASH")
df_comm['FIXED_ISSUES'].replace(np.nan, 0, inplace=True)
szz_inducing.set_index("COMMIT_HASH", inplace=True)
df_comm = df_comm.join(szz_inducing, on="COMMIT_HASH")
df_comm['INDUCED_ISSUES'].replace(np.nan, 0, inplace=True)
df_comm.reset_index(level=0, inplace=True)

print(f"done ({round(time.time()-start, 2)}s)")

## Save resulting dataframe as CSV ##
print("Saving dataframe as CSV")
df_comm.to_csv(datapath + "commit-level_dataframe.csv", index=False)

print(f"Data preparation automated pipeline finished successfully in {round(time.time()-pipeline_start, 2)} seconds.")