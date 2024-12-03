#!/usr/bin/env python

#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
from statistics import mean
from statistics import stdev as sd
import statsmodels.formula.api as smf
import datetime

#%%
datadir = "../tmp/data_281024/final/comp_constraint_0.7"
tardir = "../tmp/data_311024/final/novelty"
if not os.path.exists(tardir):
    os.makedirs(tardir)

print(f"{datetime.datetime.now()}: Started Reading CSV...", flush = True)
data = pd.read_csv(f"{datadir}/full_exp_bit.csv")
data = data[['time', 'landscape', 'exp', 'cf', 'human_knw', 'overlap', 'k']]
data = data.groupby(['landscape', 'time']).agg({
    'human_knw': 'mean',
    'overlap': 'mean',
    'k': 'mean',
    'exp': pd.Series.nunique,
    'cf': pd.Series.nunique
}).reset_index()

data['value_added'] = data['exp'] / data['cf']
data['knw_bins'] = pd.cut(data['human_knw'], bins=[0, 4, 5, 6, 7, 8, 11], labels = ['0-4', '4-5', '5-6', '6-7', '7-8', '8-11'])

#across time - knw + overlap
df1 = data.groupby(['k', 'time', 'knw_bins'])['value_added'].mean().reset_index()
print(f"{datetime.datetime.now()}: Plotting Part 1", flush = True)
for k in df1['k'].unique():
    df = df1[df1['k'] == k]
    grouped = df.groupby('knw_bins')
    for name, group in grouped:
        plt.plot(group['time'], group['value_added'], label=f'Human Knowledge - {name}')
    plt.ylabel('Average Value-added Novelty')
    plt.xlabel('Iteration')
    plt.title(f'Average Value-added Novelty Over Time Under Different Levels of Human Knowledge, K = {k}')
    plt.legend()
    plt.savefig(f"{tardir}/part1_{k}.png")
    plt.clf()

df2 = data.groupby(['k', 'time', 'overlap'])['value_added'].mean().reset_index()
print(f"{datetime.datetime.now()}: Plotting Part 2", flush = True)
for k in df2['k'].unique():
    df = df2[df2['k'] == k]
    grouped = df.groupby('overlap')
    for name, group in grouped:
        plt.plot(group['time'], group['value_added'], label=f'Overlap - {name}')
    plt.ylabel('Average Value-added Novelty')
    plt.xlabel('Iteration')
    plt.title(f'Average Value-added Novelty Over Time Under Different Levels of Overlap, K = {k}')
    plt.legend()
    plt.savefig(f"{tardir}/part2_{k}.png")
    plt.clf()

#Effect of Knw, controlled for overlap
df3 = data.groupby(['k', 'time', 'overlap', 'knw_bins'])['value_added'].mean().reset_index()
print(f"{datetime.datetime.now()}: Plotting Part 3", flush = True)
for t in [5, 10, 20, 99]:
    inter = df3[df3['time'] == t]
    for k in inter['k'].unique():
        inter2 = inter[inter['k'] == k]
        grouped = inter2.groupby('overlap')
        for name, group in grouped:
            plt.plot(group['knw_bins'], group['value_added'], label = f'Overlap - {name}')
        plt.ylabel('Average Value-added Novelty')
        plt.xlabel('Human Knowledge Level')
        plt.title('Average Value-added Novelty Over Human Knowledge Level Under Different Overlaps, t = {t}, K = {k}')
        plt.legend()
        plt.savefig(f'{tardir}/part3_k{k}_t{t}.png')
        plt.clf()

#Effect of overlap, controlled for knw
print(f"{datetime.datetime.now()}: Plotting Part 4", flush = True)
for t in [5, 10, 20, 99]:
    inter = df3[df3['time'] == t]
    for k in inter['k'].unique():
        inter2 = inter[inter['k'] == k]
        print(inter2.head(), flush = True)
        grouped = inter2.groupby('knw_bins')
        for name, group in grouped:
            plt.plot(group['overlap'], group['value_added'], label = f'Human Knowledge - {name}')
        plt.ylabel('Average Value-added Novelty')
        plt.xlabel('Overlap')
        plt.title('Average Value-added Novelty Over Overlap Under Different Levels of Human Knowledge, t = {t}, K = {k}')
        plt.legend()
        plt.savefig(f'{tardir}/part4_k{k}_t{t}.png')
        plt.clf()