#!/usr/bin/env python

#%%
import pip
def install_package(package):
    pip.main(['install', package])

install_package('patsy')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statistics import mean
from statistics import stdev as sd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrix
import datetime

#%%
datadir = "../tmp/data_281024/final/comp_constraint_0.7"
tardir = "../tmp/data_311024/final/fitness_lr"
if not os.path.exists(tardir):
    os.makedirs(tardir)

print(f"{datetime.datetime.now()}: Started Reading CSV...", flush = True)
data = pd.read_csv(f"{datadir}/full_exp_fit.csv")
data['value_added'] = data['exp'] - data['cf']
#print(f"{datetime.datetime.now()}: First CSV Done...")
#dataCF = pd.read_csv(f"{datadir}/full_cf_fit.csv", nrows = 10000)
#print(f"{datetime.datetime.now()}: Done Reading CSV...")
print("READ DONE", flush = True)

def linreg_at_t(time, data):
    model_data = data[data['time'] == time][['time', 'run', 'landscape', 'human_knw', 'value_added', 'overlap', 'k']]
    
    #demean
    model_data['overlap'] = model_data['overlap'] - model_data['overlap'].mean()
    #model_data['human_knw'] = model_data['human_knw'] - model_data['human_knw'].mean()
    print(model_data.head(), flush = True)

    #run model
    model = smf.ols('value_added ~ overlap + I(overlap**2) + C(human_knw) + overlap * C(human_knw) + C(k) + C(k)*overlap + C(k)*C(human_knw) + C(human_knw)*overlap*C(k)', data=model_data).fit(cov_type='HC3')
    summ = model.summary()

    with open(f'{tardir}/model_summary_t{time}_model.html', 'w') as f:
        f.write(summ.as_html())
    
    #check VIF
    design_matrix = dmatrix('overlap + I(overlap**2) + C(human_knw) + overlap * C(human_knw) + C(k) + C(k)*overlap + C(k)*C(human_knw) + C(human_knw)*overlap*C(k)', 
                        data=model_data, 
                        return_type='dataframe')
    vif_data = pd.DataFrame()
    vif_data["feature"] = design_matrix.columns
    vif_data["VIF"] = [variance_inflation_factor(design_matrix.values, i) for i in range(design_matrix.shape[1])]

    vif_data.to_csv(f'{tardir}/vif_t{time}.csv')

    #plot fit
    model_data['predicted_value_added'] = model.predict(model_data)

    plt.scatter(model_data['overlap'], model_data['value_added'], color='blue', label='Actual data', alpha=0.6)

    sorted_data = model_data.sort_values(by='overlap')
    plt.plot(sorted_data['overlap'], sorted_data['predicted_value_added'], color='red', label='Model fit')

    # Labels and legend
    plt.xlabel('Overlap')
    plt.ylabel('Value Added')
    plt.title('Value Added vs. Overlap with Model Fit')
    plt.legend()
    plt.savefig(f'{tardir}/model_plot_t{time}.png')
    plt.clf()

    return summ

#%%
print(f"{datetime.datetime.now()}: Starting First Linreg", flush = True)
print(linreg_at_t(5, data))

print(f"{datetime.datetime.now()}: Starting Second Linreg", flush = True)
print(linreg_at_t(10, data))

print(f"{datetime.datetime.now()}: Starting Third Linreg", flush = True)
print(linreg_at_t(20, data))

print(f"{datetime.datetime.now()}: Starting Fourth Linreg", flush = True)
print(linreg_at_t(99, data))