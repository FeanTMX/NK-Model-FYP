#!/usr/bin/env python
#this takes in raw data from the model and does the transformations from bit to novelty, time-tagged metrics and plots

#import
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import cycler
from statistics import mean
from statistics import stdev as sd

datadir_base = "../tmp/data_290924"
resultsdir_base = "../tmp/results_300924_updated_v2"
sample = 'run_0'

def hamming_dist(bit1, bit2):
    return sum(c1 != c2 for c1, c2 in zip(bit1, bit2))

def subtract_lists(lst1, lst2):
    #lst1 - lst2
    result = []
    for i in range(len(lst1)):
        result.append(lst1[i] - lst2[i])
    return result

def bit_to_novelty(df):
    #takes in a df of bits, where the columns are the runs and rows are the iterations
    #then calculates the novelty by calculating distance from original position
    df_novelty = pd.DataFrame()
    ori_poss = df.loc[0]
    for i in range(10000):
        ori_pos = ori_poss[i]
        col = "run_" + str(i)
        df_novelty[col] = df[col].apply(lambda x: hamming_dist(x, ori_pos))
    return df_novelty
        
def fitness_to_meaned_usefulness(file):
    #transforms csv_file of fitness to a list of meaned usefulness
    #the fitness file shows how the fitness progress over 100 iterations in 50 runs
    df = pd.read_csv(file)
    return df.mean(axis = 1).to_list()

def fitness_to_final_usefulness(file):
    #transforms csv_file of fitness to the final usefulness values
    #the fitness file shows how the fitness progress over 100 iterations in 50 runs
    df = pd.read_csv(file)
    return df.loc[99].to_list()[1:]

def bit_to_final_novelty(file):
    #transforms csv_file of bits to the final novelty values
    #the bit file shows how the bits progress over 100 iterations in 50 runs
    df = pd.read_csv(file)
    final_poss = df.loc[99].to_list()[1:]
    ori_poss = df.loc[0].to_list()[1:]
    result = []
    for i in range(len(final_poss)):
        result.append(hamming_dist(final_poss[i], ori_poss[i]))
    return result

results = {
    'K': [],
    'Understanding Capability': [],
    'Initial UHK': [],
    'Human Knowledge': [],
    'Run': [],
    'Final Usefulness': [],
    'Final Usefulness - CF': [],
    'Final Usefulness - Value-added': [],
    'Final Novelty': [],
    'Final Novelty - CF': [],
    'Final Novelty - Value-added': []
}

for i in [10, 15, 20, 25, 50]:
    results[f'Usefulness, t-{i}'] = []
    results[f'Usefulness - CF, t-{i}'] = []
    results[f'Novelty, t-{i}'] = []
    results[f'Novelty - CF, t-{i}'] = []

for initial_uhk in [0, 2, 4, 6]:
    for undst_cap in [0.2, 0.4, 0.6, 0.8, 0.9, 1]:
        for k in [0, 2, 4, 6, 8, 11]:
            datadir = f'{datadir_base}/initial_uhk_{initial_uhk}/undst_cap_{undst_cap}/k{k}/'
            resultsdir = f'{resultsdir_base}/initial_uhk_{initial_uhk}/undst_cap_{undst_cap}/k{k}/'
            if not os.path.exists(resultsdir):
                os.makedirs(resultsdir)
            #AI - CF data
            fit_mean_d = {}
            fit_final_d = {}
            novelty_mean_d = {}
            novelty_final_d = {}

            #AI data
            fit_mean_d_ai = {}
            fit_final_d_ai = {}
            novelty_mean_d_ai = {}
            novelty_final_d_ai = {}

            #cf data
            fit_mean_d_cf = {}
            fit_final_d_cf = {}
            novelty_mean_d_cf = {}
            novelty_final_d_cf = {}

            #sampling
            fit_sample = {}
            fit_sample_cf = {}
            novelty_sample = {}
            novelty_sample_cf = {}

            #get arguments for error plot
            y_fit = []
            x_fit = []
            e_fit = []
            
            y_nov = []
            x_nov = []
            e_nov = []

            
            for knw in range(6, 12):
                #read data
                fit = datadir + f"exp_fit_human_knw_{knw}.csv"
                fit_cf = datadir + f"cf_fit_human_knw_{knw}.csv"
                bit = datadir + f"exp_bit_human_knw_{knw}.csv"
                bit_cf = datadir + f"cf_bit_human_knw_{knw}.csv"

                fit_df = pd.read_csv(fit, index_col = 0)
                fit_cf_df = pd.read_csv(fit_cf, index_col = 0)
                bit_df = pd.read_csv(bit, index_col = 0)
                bit_cf_df = pd.read_csv(bit_cf, index_col = 0)

                #create a novelty df based on bit
                novelty_df = bit_to_novelty(bit_df)
                novelty_cf_df = bit_to_novelty(bit_cf_df)

                #take the mean across trials
                fit_mean = fit_df.mean(axis = 1).to_list()
                fit_cf_mean = fit_cf_df.mean(axis = 1).to_list()
                novelty_mean = novelty_df.mean(axis = 1).to_list()
                novelty_cf_mean = novelty_cf_df.mean(axis = 1).to_list()

                #take final values
                fit_final = fit_df.loc[99].to_list()
                fit_cf_final = fit_cf_df.loc[99].to_list()
                novelty_final = novelty_df.loc[99].to_list()
                novelty_cf_final = novelty_cf_df.loc[99].to_list()
                
                #take AI - cf for value-added
                fit_diff_mean = subtract_lists(fit_mean, fit_cf_mean)
                fit_diff_final = subtract_lists(fit_final, fit_cf_final)
                novelty_diff_mean = subtract_lists(novelty_mean, novelty_cf_mean)
                novelty_diff_final = subtract_lists(novelty_final, novelty_cf_final)
                
                #add to df
                fit_mean_d[f"human_knw_{knw}"] = fit_diff_mean
                fit_final_d[f"human_knw_{knw}"] = fit_diff_final
                novelty_mean_d[f"human_knw_{knw}"] = novelty_diff_mean
                novelty_final_d[f"human_knw_{knw}"] = novelty_diff_final

                fit_mean_d_ai[f"human_knw_{knw}"] = fit_mean
                fit_final_d_ai[f"human_knw_{knw}"] = fit_final
                novelty_mean_d_ai[f"human_knw_{knw}"] = novelty_mean
                novelty_final_d_ai[f"human_knw_{knw}"] = novelty_final

                fit_mean_d_cf[f"human_knw_{knw}"] = fit_cf_mean
                fit_final_d_cf[f"human_knw_{knw}"] = fit_cf_final
                novelty_mean_d_cf[f"human_knw_{knw}"] = novelty_cf_mean
                novelty_final_d_cf[f"human_knw_{knw}"] = novelty_cf_final

                fit_sample[f"human_knw_{knw}"] = fit_df[sample]
                fit_sample_cf[f"human_knw_{knw}"] = fit_cf_df[sample]
                novelty_sample[f"human_knw_{knw}"] = novelty_df[sample]
                novelty_sample_cf[f"human_knw_{knw}"] = novelty_cf_df[sample]
                
                y_fit.append(mean(fit_diff_final))
                x_fit.append(knw)
                e_fit.append(sd(fit_diff_final))
                
                y_nov.append(mean(novelty_diff_final))
                x_nov.append(knw)
                e_nov.append(sd(fit_diff_final))

                #append to final dictionary
  

                results['K'] += [k] * len(fit_final)
                results['Understanding Capability'] += [undst_cap] * len(fit_final)
                results['Initial UHK'] += [initial_uhk] * len(fit_final)
                results['Human Knowledge'] += [knw] * len(fit_final)
                results['Run'] += [f'run_{i}' for i in range(len(fit_final))]
                results['Final Usefulness'] += fit_final
                results['Final Usefulness - CF'] += fit_cf_final
                results['Final Usefulness - Value-added'] += fit_diff_final
                results['Final Novelty'] += novelty_final
                results['Final Novelty - CF'] += novelty_cf_final
                results['Final Novelty - Value-added'] += novelty_diff_final
                
                for i in [10, 15, 20, 25, 50]:
                    results[f'Usefulness, t-{i}'] += fit_df.loc[i-1].to_list()
                    results[f'Usefulness - CF, t-{i}'] += fit_cf_df.loc[i-1].to_list()
                    results[f'Novelty, t-{i}'] += novelty_df.loc[i-1].to_list()
                    results[f'Novelty - CF, t-{i}'] += novelty_cf_df.loc[i-1].to_list()


            fit_mean_df = pd.DataFrame(fit_mean_d)
            fit_final_df = pd.DataFrame(fit_final_d)
            novelty_mean_df = pd.DataFrame(novelty_mean_d)
            novelty_final_df = pd.DataFrame(novelty_final_d)

            fit_mean_df_ai = pd.DataFrame(fit_mean_d_ai)
            fit_final_df_ai = pd.DataFrame(fit_final_d_ai)
            novelty_mean_df_ai = pd.DataFrame(novelty_mean_d_ai)
            novelty_final_df_ai = pd.DataFrame(novelty_final_d_ai)

            fit_mean_df_cf = pd.DataFrame(fit_mean_d_cf)
            fit_final_df_cf = pd.DataFrame(fit_final_d_cf)
            novelty_mean_df_cf = pd.DataFrame(novelty_mean_d_cf)
            novelty_final_df_cf = pd.DataFrame(novelty_final_d_cf)

            fit_sample_df_ai = pd.DataFrame(fit_sample)
            fit_sample_df_cf = pd.DataFrame(fit_sample_cf)
            novelty_sample_df_ai = pd.DataFrame(novelty_sample)
            novelty_sample_df_cf = pd.DataFrame(novelty_sample_cf)

            #plot mean fitness overtime
            for col in fit_mean_df:
                plt.plot(fit_mean_df[col], label = col)
            plt.legend()
            plt.ylabel('Usefulness')
            plt.xlabel('Iteration')
            plt.title(f'Usefuless over time Under Different Levels of Human Knowledge, K = {k}')
            plt.savefig(resultsdir + f'useful_v_time_k_{k}')
            plt.clf()
            
            #plot mean novelty overtime
            for col in novelty_mean_df:
                plt.plot(novelty_mean_df[col], label = col)
            plt.legend()
            plt.ylabel('Novelty')
            plt.xlabel('Iteration')
            plt.title(f'Novelty over time Under Different Levels of Human Knowledge, K = {k}')
            plt.savefig(resultsdir + f'novel_v_time_k_{k}')
            plt.clf()
            
            #plot final usefulness v human knowledge
            plt.errorbar(x_fit, y_fit, e_fit, marker='^')
            plt.ylabel('Final Usefulness')
            plt.xlabel('Human Knowledge')
            plt.title(f'Final Usefulness over Human Knowledge, K = {k}')
            plt.savefig(resultsdir + f'useful_v_knw_{k}')
            plt.clf()
            
            #plot final novelty v human knowledge
            plt.errorbar(x_nov, y_nov, e_nov, marker='^')
            plt.ylabel('Final Novelty')
            plt.xlabel('Human Knowledge')
            plt.title(f'Final Novelty over Human Knowledge, K = {k}')
            plt.savefig(resultsdir + f'nov_v_knw_{k}')
            plt.clf()

            #plot mean fitness overtime, AI only
            for col in fit_mean_df_ai:
                plt.plot(fit_mean_df_ai[col], label = col)
            plt.legend()
            plt.ylabel('Usefulness')
            plt.xlabel('Iteration')
            plt.title(f'Usefuless over time Under Different Levels of Human Knowledge, K = {k}, AI only')
            plt.savefig(resultsdir + f'ai_useful_v_time_k_{k}')
            plt.clf()

            #plot mean novelty overtime, AI only
            for col in novelty_mean_df_ai:
                plt.plot(novelty_mean_df_ai[col], label = col)
            plt.legend()
            plt.ylabel('Novelty')
            plt.xlabel('Iteration')
            plt.title(f'Novelty over time Under Different Levels of Human Knowledge, K = {k}, AI only')
            plt.savefig(resultsdir + f'ai_novel_v_time_k_{k}')
            plt.clf()

            #plot mean fitness overtime, cf only
            for col in fit_mean_df_cf:
                plt.plot(fit_mean_df_cf[col], label = col)
            plt.legend()
            plt.ylabel('Usefulness')
            plt.xlabel('Iteration')
            plt.title(f'Usefuless over time Under Different Levels of Human Knowledge, K = {k}, Counterfactual')
            plt.savefig(resultsdir + f'cf_useful_v_time_k_{k}')
            plt.clf()

            #plot mean novelty overtime, cf only
            for col in novelty_mean_df_cf:
                plt.plot(novelty_mean_df_cf[col], label = col)
            plt.legend()
            plt.ylabel('Novelty')
            plt.xlabel('Iteration')
            plt.title(f'Novelty over time Under Different Levels of Human Knowledge, K = {k}, Counterfactual')
            plt.savefig(resultsdir + f'cf_novel_v_time_k_{k}')
            plt.clf()

            #plot final fitness, AI v cf
            plt.plot(fit_final_df_ai.mean(), label = 'AI')
            plt.plot(fit_final_df_cf.mean(), label = 'Counterfactual')
            plt.ylabel('Final Usefulness')
            plt.xlabel('Human Knowledge')
            plt.title(f'Final Usefulness over Human Knowledge, K = {k}, AI v Counterfactual')
            plt.savefig(resultsdir + f'ai_cf_useful_v_knw_{k}')
            plt.clf()

            #plot final novelty, AI v cf
            plt.plot(novelty_final_df_ai.mean(), label = 'AI')
            plt.plot(novelty_final_df_cf.mean(), label = 'Counterfactual')
            plt.ylabel('Final Novelty')
            plt.xlabel('Human Knowledge')
            plt.title(f'Final Novelty over Human Knowledge, K = {k}, AI v Counterfactual')
            plt.savefig(resultsdir + f'ai_cf_nov_v_knw_{k}')
            plt.clf()

            #plot samples
            for knw in range(6, 12):
                col = f'human_knw_{knw}'
                ai = fit_sample_df_ai[col]
                cf = fit_sample_df_cf[col]
                plt.plot(ai, label = 'AI')
                plt.plot(cf, label = 'Counterfactual')
                plt.legend()
                plt.ylabel('Usefulness')
                plt.xlabel('Iteration')
                plt.title(f'Usefuless over time - sample, K = {k}, human_knowledge = {knw}')
                plt.savefig(resultsdir + f'sample_useful_v_time_k_{k}_HmnKnw_{knw}')
                plt.clf()
                
            for knw in range(6, 12):
                col = f'human_knw_{knw}'
                ai = novelty_sample_df_ai[col]
                cf = novelty_sample_df_cf[col]
                plt.plot(ai, label = 'AI')
                plt.plot(cf, label = 'Counterfactual')
                plt.legend()
                plt.ylabel('Novelty')
                plt.xlabel('Iteration')
                plt.title(f'Novelty over time - sample, K = {k}, human_knowledge = {knw}')
                plt.savefig(resultsdir + f'sample_novel_v_time_k_{k}_HmnKnw_{knw}')
                plt.clf()

output = pd.DataFrame(results)
for i in [10, 15, 20, 25, 50]:
    results[f'Usefulness - Value Added, t-{i}'] = results[f'Usefulness, t-{i}'] - results[f'Usefulness - CF, t-{i}']
    results[f'Novelty - Value Added, t-{i}'] = results[f'Novelty, t-{i}'] - results[f'Novelty - CF, t-{i}']
output.to_csv(resultsdir_base + '/results.csv')
