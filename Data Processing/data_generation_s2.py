#!/usr/bin/env python

#%%
import pandas as pd
import os
import multiprocessing
import datetime

#%%
datadir_base = "../tmp/data_281024/comp_constraint_0.7/clean2"
tardir_base = "../tmp/data_281024/final/comp_constraint_0.7"

overlaps = [0, 1, 2, 3, 4, 5, 6]
k = [0, 2, 4, 6, 8, 11]
cnt = 0

flag = False
for ol in overlaps:
    for k_curr in k:
        print(f"Point 1 - {datetime.datetime.now()}: Overlap: {ol}, K: {k_curr}", flush = True)
        #inter_cf_bit = pd.read_csv(f'{datadir_base}/overlap_{ol}/k{k_curr}/cf_bit.csv')
        #inter_cf_fit = pd.read_csv(f'{datadir_base}/overlap_{ol}/k{k_curr}/cf_fit.csv')
        #inter_exp_bit = pd.read_csv(f'{datadir_base}/overlap_{ol}/k{k_curr}/exp_bit.csv')
        inter_exp_fit = pd.read_csv(f'{datadir_base}/overlap_{ol}/k{k_curr}/exp_fit.csv')

        #cause its diff landscape for each experiment
        #inter_cf_bit['landscape'] = inter_cf_bit['landscape'].apply(lambda x: f'{x}_{str(cnt)}')
        #inter_cf_fit['landscape'] = inter_cf_fit['landscape'].apply(lambda x: f'{x}_{str(cnt)}')
        #inter_exp_bit['landscape'] = inter_exp_bit['landscape'].apply(lambda x: f'{x}_{str(cnt)}')
        inter_exp_fit['landscape'] = inter_exp_fit['landscape'].apply(lambda x: f'{x}_{str(cnt)}')

        print(f"Point 2 - {datetime.datetime.now()}: Overlap: {ol}, K: {k_curr}", flush = True)

        if flag:
            #output_cf_bit = pd.concat([output_cf_bit, inter_cf_bit])
            #output_cf_fit = pd.concat([output_cf_fit, inter_cf_fit])
            #output_exp_bit = pd.concat([output_exp_bit, inter_exp_bit])
            output_exp_fit = pd.concat([output_exp_fit, inter_exp_fit])
        else:
            flag = True
            #output_cf_bit = inter_cf_bit
            #output_cf_fit = inter_cf_fit
            #output_exp_bit = inter_exp_bit
            output_exp_fit = inter_exp_fit
        
        print(f"Point 3 - {datetime.datetime.now()}: Overlap: {ol}, K: {k_curr}", flush = True)
        cnt += 1
        #print(output_exp_bit.head())
        #print(output_exp_bit.shape)

try:
    os.makedirs(f'{tardir_base}/', exist_ok=True)
    output_exp_fit.to_csv(f'{tardir_base}/full_exp_fit.csv')
    #output_exp_bit.to_csv(f'{tardir_base}/full_exp_bit.csv')
    #output_cf_fit.to_csv(f'{tardir_base}/full_cf_fit.csv')
    #output_cf_bit.to_csv(f'{tardir_base}/full_cf_bit.csv')
    print("write done")
except Exception as e:
    print(f"error: {e}")
