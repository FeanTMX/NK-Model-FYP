#!/usr/bin/env python

import pip

def install_package(package):
    pip.main(['install', package])

install_package('pandas')
install_package('numpy')

#%%
import pandas as pd
import os
import multiprocessing
import datetime

#%%
def convert_to_long(df):
    print('test')
    #converts raw data to long
    df = df.rename(columns = {"Unnamed: 0": "time"})
    #print(df.head(), flush = True)
    melted = pd.melt(df, id_vars = 'time', var_name = 'name', value_name = 'value')
    #print('melted',melted.head(), flush=True)
    melted[['temp', 'suffix']] = melted['name'].str.split('-', expand=True)
    long = melted.pivot(index=['time', 'temp'], columns = 'suffix', values = 'value').reset_index()
    
    #print(long.head(), flush = True)
    #print('long', long.head(), flush = True)
    long['run'] = long['temp'].apply(lambda x: x.split('_')[1])
    long['landscape'] = long['temp'].apply(lambda x: f'{x.split("_")[3]}')
    long['human_knw'] = long['temp'].apply(lambda x: x.split('_')[-1])
    return long[['time', 'run', 'landscape', 'human_knw', 'exp', 'cf']]

#%%
def job(ol, k, datadir_base, tardir_base):
    for k_curr in k:
        print(datetime.datetime.now())
        print(f'{datetime.datetime.now()}: Overlap: {ol}, K: {k_curr}', flush = True)
        try:
            os.makedirs(f'{tardir_base}/overlap_{ol}/k{k_curr}/', exist_ok=True)
            #read
            #df_cf_fit = pd.read_csv(f'{datadir_base}/overlap_{ol}/k{k_curr}/cf_fit.csv')
            #df_cf_bit = pd.read_csv(f'{datadir_base}/overlap_{ol}/k{k_curr}/cf_bit.csv')
            #print('start read', flush = True)
            df_exp_fit = pd.read_csv(f'{datadir_base}/overlap_{ol}/k{k_curr}/exp_fit.csv')
            df_exp_bit = pd.read_csv(f'{datadir_base}/overlap_{ol}/k{k_curr}/exp_bit.csv')
            print('read done', flush = True)
        
            #df_cf_bit.columns = df_exp_fit.columns
            #df_cf_fit.columns = df_exp_fit.columns
            #df_exp_bit.columns = df_exp_fit.columns

            #transform
            #long_cf_bit = convert_to_long(df_cf_bit)
            #long_cf_fit = convert_to_long(df_cf_fit)
            long_exp_bit = convert_to_long(df_exp_bit)
            long_exp_fit = convert_to_long(df_exp_fit)

            #add
            #long_cf_bit['overlap'] = ol
            #long_cf_bit['k'] = k_curr
            #long_cf_fit['overlap'] = ol
            #long_cf_fit['k'] = k_curr
            long_exp_fit['overlap'] = ol
            long_exp_fit['k'] = k_curr
            long_exp_bit['overlap'] = ol
            long_exp_bit['k'] = k_curr
            #print(long_exp_fit.columns, flush = True)
            #print(long_exp_fit.head(), flush=True)
            #print(long_exp_bit.columns, flush = True)
            #print(long_exp_bit.head(), flush=True)
            

            #write
            #long_cf_bit.to_csv(f'{tardir_base}/overlap_{ol}/k{k_curr}/cf_bit.csv')
            #long_cf_fit.to_csv(f'{tardir_base}/overlap_{ol}/k{k_curr}/cf_fit.csv')
            long_exp_bit.to_csv(f'{tardir_base}/overlap_{ol}/k{k_curr}/exp_bit.csv')
            long_exp_fit.to_csv(f'{tardir_base}/overlap_{ol}/k{k_curr}/exp_fit.csv')
            print(f'{datetime.datetime.now()}: WRITE DONE - Overlap: {ol}, K: {k_curr}', flush = True)
        except Exception as e:
            print(f"data reading failed - overlap_{ol}, k_{k_curr}")
            print(f"Error: {e}")

print('before main')

if __name__ == '__main__':
    print('in main')
    datadir_base = "../tmp/data_281024/comp_constraint_0.7"
    tardir_base = "../tmp/data_281024/comp_constraint_0.7/clean2"
    overlaps = [0, 1, 2, 3, 4, 5, 6]
    k = [0, 2, 4, 6, 8, 11]

    print('in main 2')

    with multiprocessing.Pool() as pool:
        print("test-2")
        job_args = [
            (ol, k, datadir_base, tardir_base)
            for ol in overlaps
        ]
        pool.starmap(job, job_args)
        pool.close()
        pool.join()
