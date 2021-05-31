# SEQUENTIAL DOMAIN ADAPTATION by SYNTHESIZING DISTRIBUTIONALLY ROBUST EXPERTS
Bahar Taskesen, Man-Chung Yue, Jose Blanchet, Daniel Kuhn and Viet Anh Nguyen

## Introduction
Least squares estimators, when trained on a few target domain samples, may predict poorly. 
Supervised domain adaptation aims to improve the predictive accuracy by exploiting additional labeled training samples from a source distribution that is close to the target distribution. 
Given available data, we investigate novel strategies to synthesize a family of least squares estimator experts that are robust with regard to moment conditions.
    
## Quick Start
MATLAB version should be at least 2020a.
This repository contains of distributionally robust experts for supervised domain adaptation presented in the paper. 
Install YALMIP from https://yalmip.github.io/tutorial/installation/ and MOSEK from https://docs.mosek.com/9.2/install/installation.html by following the instructions.

## Numerical Experiments 
The results in the numerical experiments section (Table 1, and Figure 4) are obtained by running run_main.m script. 
All the supplementary functions are placed under the **src** folder.
#### Datasets
Due to capacity limits the datasets are not available under this repository. However, all datasets are available publicly at
- Uber&Lyft: https://www.kaggle.com/brllrb/uber-and-lyft-dataset-boston-ma.
Should be placed as "./data/kaggle/uber/rideshare_kaggle.csv"

- US Births (2018): https://www.kaggle.com/des137/us-births-2018.
Should be placed as "'./data/kaggle/birth_data/USbirths_2018.csv'"

- Life Expectancy: https://www.kaggle.com/kumarajarshi/life-expectancy-who
Should be placed as  "./data/LifeExpectancyDataset.csv"

- House Prices in King Country: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
Should be placed as "./data/houses/train.csv"

- California Housing Prices: https://www.kaggle.com/camnugent/california-housing-pricesand
Should be placed as "./data/california_housing.csv"


#### Saved numerical results
The workspaces of the experiment results in Secion 6, in particular Table 1 and Figure 4 as well as Figure A.1. in the appendix are placed under "./paper results".
For each dataset the corresponding figure is obtained by running ./paper results/plotting.m to obtain the figures in the paper with the data_set parameter set accordingly. 
The values in the table are obtained by running ./paper results/create_table_values.m. Detailed explanations on how to run these codes are provided at the begining of the scripts.










