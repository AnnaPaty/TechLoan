TechLoan
==============================

The main motivation for this project is to gain valuable insight into the usual management of software engineering projects in order to improve the performance of development teams based on the detection of recurrent flaws in the analyzed projects from the TDD.

Specifically, the project goal is to categorize developers by how good they are and the experience they have in solving errors. This way, the insights could be used to better balance the development squads in terms of having developers with different expertise. Another use of these insights could be to better assign the error solving tasks to the right developers.


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump, just the .zip is included to space limits reasons.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── reports            <- Generated analysis as PDF.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        |   ├── __init__.py  
        │   └── data_preparation.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        |   ├── __init__.py
        │   └── commit_to_committer.py
        │
        └── models         <- Scripts to train models and then use trained models to make predictions
            ├── __init__.py
            ├── auto_squading.py
            └── modelling.py

Below is an illustration that exemplifies the usage of this template in our project:

![](https://github.com/AnnaPaty/TechLoan/blob/main/reports/figures/flow-diagram.png?raw=true)

Steps to reproduce the results
--------
1. Install requirements:
````
pip install -r requirements.txt
````
2. Execute data processing:
````
python src/data/data_preparation.py
python src/features/commit_to_committer.py
````
3. Train and save the model:
````
python src/models/modelling.py
````
4. Generate balanced teams:
````
python src/models/auto_squading.py
````
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
