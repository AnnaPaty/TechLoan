import numpy as np
import pandas as pd
import time

# Models
import modelling as mod

# Save squads
import csv

# Find the data
import os
root = os.getcwd().split("TechLoan")[0]+"TechLoan/data/"
df = pd.read_csv(root+"processed/committer-level_dataframe.csv")


def check_input(mode, prompt):
    if mode == "ndevs":
        str_devsXsquad = input(prompt)   # number of developers in each squad
        try:
            devsXsquad = int(str_devsXsquad)
            if devsXsquad > 3:
                return devsXsquad
            raise Exception
        except:
            print("The number of developers per squad must be a number higher than 3.")
            return None

    if mode == "y/n":
        str_yn  = input(prompt)
        if type(str_yn) is str and str_yn in ["y", "n"]:
            return str_yn == "y"
        else:
            print("Type 'y' or 'n'.")
            return None


def read_input():
    # Read number of devs per squad
    while True:
        devsXsquad = check_input(mode="ndevs", prompt="Number of developers per squad: ")
        if devsXsquad is not None:
            break
    
    # Read save option
    while True:
        bool_save = check_input(mode="y/n", prompt="Do you want to save the squads to a CSV file (y/n)? ")
        if bool_save is not None:
            break

    # Read print option
    while True:
        bool_print = check_input(mode="y/n", prompt="Do you want to print the squads? (y/n) ")
        if bool_print is not None:
            break

    return devsXsquad, bool_save, bool_print


# Users have to explore the clusters to determine what experience level is represented in each cluster.
# Levels are ordered by the cluster represented ('senior' is represented by cluster 0).
clust2XP = ["senior", "experienced", "newbie"]

# Get input from user
devsXsquad, bool_save, bool_print = read_input()


def parameter_error(df, devsXsquad, clust2XP):
    error = False
    if type(df) != pd.core.frame.DataFrame:
        print("ERROR: Parameter 'df' needs to be a pandas dataframe.")
        error = True
    elif type(devsXsquad) != int or devsXsquad < 3:
        print("ERROR: Check parameter 'devsXsquad'.")
        error = True
    elif len(clust2XP) != 3 or not ('senior' in clust2XP and
                                    'experienced' in clust2XP and
                                    'newbie' in clust2XP):
        print("ERROR: Check parameter 'clust2XP'.")
        error = True
    return error


def print_squads(squads):
    for i,s in enumerate(squads):
        print(f"Squad #{i}:")
        for lvl in s.keys():
            print(" "*5 + f"{lvl}:")
            for dev in s[lvl]:
                print(" "*10 + dev)


def proportional_autosquad(df, devsXsquad, clustToXP, bool_print=False):
    if parameter_error(df, devsXsquad, clustToXP):
        return
    # Clusterize developers using the techniques we found to give the best results
    devs = np.array(df["COMMITTER"])
    embedded = mod.umap.UMAP().fit_transform(df.loc[:, df.columns != 'COMMITTER'])
    clust = mod.hieragglo(embedded, linkage="complete", criteria="n_clusters", parameter=3)

    # Form squads based on the proportion of each cluster so that squads are balanced
    grouped_devs = {clust2XP[i]:devs[[bool(c == i) for c in clust]] for i in range(3)}
    prop = {clust2XP[i]:round(float(sum(clust == i)/len(df))*devsXsquad) for i in range(3)}
    nsquads = int(min([len(grouped_devs[lvl])/prop[lvl] for lvl in clust2XP]))
    squads = []
    for i in range(nsquads):
        squads.append({lvl:grouped_devs[lvl][i*prop[lvl]:(i+1)*prop[lvl]] for lvl in clust2XP})

    # Print the squads
    if bool_print:
        print_squads(squads)

    return squads


squad = proportional_autosquad(df, 5, ["senior","experienced","newbie"], bool_print=bool_print)


if bool_save:
    start = time.time()
    print(f"Saving squads to {root+'squads/squads.csv'}...", end=" ")
    f = open(root+'squads/squads.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(["squad","senior","experienced","newbie"])
    for id,s in enumerate(squad):
        row = [id]
        for cluster in s:
            row.append(s[cluster])
        writer.writerow(row)
    f.close()
    print(f"done ({round(time.time()-start, 2)} sec)")
