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
tardir = "../tmp/data_311024/final/fitness"
if not os.path.exists(tardir):
    os.makedirs(tardir)

print(f"{datetime.datetime.now()}: Started Reading CSV...", flush = True)
data = pd.read_csv(f"{datadir}/full_exp_fit.csv")
data['value_added'] = data['exp'] - data['cf']
#print(f"{datetime.datetime.now()}: First CSV Done...")
#dataCF = pd.read_csv(f"{datadir}/full_cf_fit.csv", nrows = 10000)
#print(f"{datetime.datetime.now()}: Done Reading CSV...")
print("READ DONE", flush = True)

#across time - knw + overlap
df2 = data.groupby(['k', 'time', 'human_knw'])['value_added'].mean().reset_index()
print(f"{datetime.datetime.now()}: Plotting Part 1", flush = True)
for k in df2['k'].unique():
    df = df2[df2['k'] == k]
    grouped = df.groupby('human_knw')
    for name, group in grouped:
        plt.plot(group['time'], group['value_added'], label=f'Human Knowledge - {name}')
    plt.ylabel('Average Value-added Usefulness')
    plt.xlabel('Iteration')
    plt.title(f'Average Value-added Usefuless Over Time Under Different Levels of Human Knowledge, K = {k}')
    plt.legend()
    plt.savefig(f"{tardir}/part1_{k}")
    plt.clf()

df3 = data.groupby(['k', 'time', 'overlap'])['value_added'].mean().reset_index()
print(f"{datetime.datetime.now()}: Plotting Part 2", flush = True)
for k in df3['k'].unique():
    df = df3[df3['k'] == k]
    grouped = df.groupby('overlap')
    for name, group in grouped:
        plt.plot(group['time'], group['value_added'], label=f'Overlap - {name}')
    plt.ylabel('Average Value-added Usefulness')
    plt.xlabel('Iteration')
    plt.title(f'Average Value-added Usefuless Over Time Under Different Levels of Overlap, K = {k}')
    plt.legend()
    plt.savefig(f"{tardir}/part2_{k}")
    plt.clf()

#Effect of Knw, controlled for overlap
df1 = data.groupby(['k', 'time', 'human_knw','overlap'])['value_added'].mean().reset_index()
print(f"{datetime.datetime.now()}: Plotting Part 3", flush = True)
for t in [5, 10, 20, 99]:
    inter = df1[df1['time'] == t]
    for k in inter['k'].unique():
        inter2 = inter[inter['k'] == k]
        grouped = inter2.groupby('overlap')
        for name, group in grouped:
            plt.plot(group['human_knw'], group['value_added'], label = f'Overlap = {name}')
        plt.ylabel('Average Value-added Usefulness')
        plt.xlabel('Human Knowledge')
        plt.title(f'Average Value-added Usefulness Over Human Knowledge Under Different Levels of Overlap at t = {t}, K = {k}')
        plt.legend()
        plt.savefig(f"{tardir}/part3_t{t}_k{k}")
        plt.clf()

#Effect of overlap, controlled for knw
print(f"{datetime.datetime.now()}: Plotting Part 4", flush = True)
for t in [5, 10, 20, 99]:
    inter = df1[df1['time'] == t]
    for k in inter['k'].unique():
        inter2 = inter[inter['k'] == k]
        grouped = inter2.groupby('human_knw')
        for name, group in grouped:
            plt.plot(group['overlap'], group['value_added'], label=f'Human Knowledge - {name}')
        plt.ylabel('Average Value-added Usefulness')
        plt.xlabel('Overlap')
        plt.title(f'Average Value-added Usefulness Over Overlap Under Different Levels of Human Knowledge at t = {t}, K = {k}')
        plt.legend()
        plt.savefig(f"{tardir}/part4_t{t}_k{k}")
        plt.clf()

#Effect of K
df4 = data.groupby(['k', 'time', 'human_knw'])['exp'].mean().reset_index()
print(f"{datetime.datetime.now()}: Plotting Part 5", flush = True)
for t in [5, 10, 20, 99]:
    inter = df4[df4['time'] == t]
    grouped = inter.groupby('human_knw')
    for name, group in grouped:
        plt.plot(group['k'], group['exp'], label=f'Human Knowledge - {name}')
    plt.ylabel('Average Usefulness')
    plt.xlabel('K')
    plt.title(f'Average Usefulness Over K Under Different Levels of Human Knowledge at t = {t}')
    plt.legend()
    plt.savefig(f"{tardir}/part5_expXk_t{t}")
    plt.clf()

df5 = data.groupby(['k', 'time', 'overlap'])['exp'].mean().reset_index()
print(f"{datetime.datetime.now()}: Plotting Part 6", flush = True)
for t in [5, 10, 20, 99]:
    inter = df5[df5['time'] == t]
    grouped = inter.groupby('overlap')
    for name, group in grouped:
        plt.plot(group['k'], group['exp'], label=f'Overlap - {name}')
    plt.ylabel('Average Usefulness')
    plt.xlabel('K')
    plt.title(f'Average Usefulness Over K Under Different Levels of Overlap at t = {t}')
    plt.legend()
    plt.savefig(f"{tardir}/part6_expXk__t{t}")
    plt.clf()

df6 = data.groupby(['k', 'time', 'human_knw'])['cf'].mean().reset_index()
print(f"{datetime.datetime.now()}: Plotting Part 5", flush = True)
for t in [5, 10, 20, 99]:
    inter = df6[df6['time'] == t]
    grouped = inter.groupby('human_knw')
    for name, group in grouped:
        plt.plot(group['k'], group['cf'], label=f'Human Knowledge - {name}')
    plt.ylabel('Average Usefulness')
    plt.xlabel('K')
    plt.title(f'Average Usefulness Over K Under Different Levels of Human Knowledge at t = {t}')
    plt.legend()
    plt.savefig(f"{tardir}/part7_cfXk_t{t}")
    plt.clf()

df7 = data.groupby(['k', 'time', 'overlap'])['cf'].mean().reset_index()
print(f"{datetime.datetime.now()}: Plotting Part 6", flush = True)
for t in [5, 10, 20, 99]:
    inter = df7[df7['time'] == t]
    grouped = inter.groupby('overlap')
    for name, group in grouped:
        plt.plot(group['k'], group['cf'], label=f'Overlap - {name}')
    plt.ylabel('Average Usefulness')
    plt.xlabel('K')
    plt.title(f'Average Usefulness Over K Under Different Levels of Overlap at t = {t}')
    plt.legend()
    plt.savefig(f"{tardir}/part8_cfXk_t{t}")
    plt.clf()