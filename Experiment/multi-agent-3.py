#!/usr/bin/env python
#%%

#installer
import pip
def install_package(package):
    pip.main(['install', package])

#install
install_package('pandas')
install_package('matplotlib')

#import
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import random as rand
import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing
import datetime

#function def
def hamming_dist(bit1, bit2):
    return sum(c1 != c2 for c1, c2 in zip(bit1, bit2))

def generate_ai_cognitive_space(ai_kb, human_kb, undst_cap):
    #takes in the ai knowledge base, human knowledge base and undst capability
    #returns the cognitive space
    #undst capability will determine how many elements of the human knowledge base can the ai 'see'
    #then, we will randomly sample that number of elements. If it not something in the AI's cognitive space already, it will be appended
    unds = int(undst_cap * len(human_kb))
    choice = list(np.random.choice(human_kb, unds, replace=False))
    return sorted(list(set(ai_kb).union(set(choice))))

class LandScape():

    def __init__(self, N, K, K_within, K_between):
        self.N = N
        self.K = K
        self.K_within = K_within
        self.K_between = K_between
        self.IM, self.IM_dic = None, None
        self.FC = None
        self.cache = {}  # the hashed dict has a higher indexing speed, which helps improve the running speed
        self.cog_cache = {}

    def create_influence_matrix(self):
        IM = np.eye(self.N)
        if self.K_within is None:
            for i in range(self.N):
                probs = [1 / (self.N - 1)] * i + [0] + [1 / (self.N - 1)] * (self.N - 1 - i)
                ids = np.random.choice(self.N, self.K, p=probs, replace=False)
                for index in ids:
                    IM[i][index] = 1
        else:
            for i in range(self.N):
                if i // (self.N // 2) < 1:
                    within = [j for j in range(self.N // 2)]
                    between = [j for j in range(self.N // 2, self.N)]
                    probs = [1 / (self.N // 2 - 1)] * i + [0] + [1 / (self.N // 2 - 1)] * (self.N // 2 - 1 - i)
                    ids_within = np.random.choice(within, self.K_within, p=probs, replace=False)
                    ids_between = np.random.choice(between, self.K_between, replace=False)
                    for index in ids_within:
                        IM[i][index] = 1
                    for index in ids_between:
                        IM[i][index] = 1

                else:
                    within = [j for j in range(self.N // 2, self.N)]
                    between = [j for j in range(self.N // 2)]
                    probs = [1 / (self.N // 2 - 1)] * (i - self.N // 2) + [0] + [1 / (self.N // 2 - 1)] * (
                                self.N - 1 - i)
                    ids_between = np.random.choice(between, self.K_between, replace=False)
                    ids_within = np.random.choice(within, self.K_within, p=probs, replace=False)
                    for index in ids_within:
                        IM[i][index] = 1
                    for index in ids_between:
                        IM[i][index] = 1

        IM_dic = defaultdict(list)
        for i in range(len(IM)):
            for j in range(len(IM[0])):
                if i == j or IM[i][j] == 0:
                    continue
                else:
                    IM_dic[i].append(j)
        self.IM, self.IM_dic = IM, IM_dic

    def create_fitness_config(self,):
        FC = defaultdict(dict)
        for row in range(len(self.IM)):

            k = int(sum(self.IM[row]))
            for i in range(pow(2, k)):
                FC[row][i] = np.random.uniform(0, 1)
        self.FC = FC


    def calculate_fitness(self, state):
        res = 0.0
        for i in range(len(state)):
            dependency = self.IM_dic[i]
            bin_index = "".join([str(state[j]) for j in dependency])
            if state[i] == 0:
                bin_index = "0" + bin_index
            else:
                bin_index = "1" + bin_index
            index = int(bin_index, 2)
            res += self.FC[i][index]
        return res / len(state)

    def store_cache(self,):
        for i in range(pow(2,self.N)):
            bit = bin(i)[2:]
            if len(bit)<self.N:
                bit = "0"*(self.N-len(bit))+bit
            state = [int(cur) for cur in bit]
            self.cache[bit] = self.calculate_fitness(state)


    def initialize(self, first_time=True, norm=True):
        if first_time:
            self.create_influence_matrix()
        self.create_fitness_config()
        self.store_cache()

        # normalization
        if norm:
            normalizor = max(self.cache.values())
            min_normalizor = min(self.cache.values())

            for k in self.cache.keys():
                self.cache[k] = (self.cache[k]-min_normalizor)/(normalizor-min_normalizor)
        self.cog_cache = {}

    def query_fitness(self, state):
        bit = "".join([str(state[i]) for i in range(len(state))])
        return self.cache[bit]

    def query_cog_fitness(self, state, knowledge_sapce):
        remainder = [cur for cur in range(self.N) if cur not in knowledge_sapce]
        regular_expression = "".join(str(state[i]) if i in knowledge_sapce else "*" for i in range(len(state)))
        if regular_expression in self.cog_cache:
            return self.cog_cache[regular_expression]

        remain_length = len(remainder)
        res = 0
        for i in range(pow(2, remain_length)):
            bit = bin(i)[2:]
            if len(bit)<remain_length:
                bit = "0"*(remain_length-len(bit))+bit
            temp_state = list(state)

            for j in range(remain_length):
                temp_state[remainder[j]] = int(bit[j])
            res+=self.query_fitness(temp_state)
        res = 1.0*res/pow(2, remain_length)
        self.cog_cache[regular_expression] = res

        return res

class Agent:
    
    def __init__(self, N, landscape, knowledge_base, acceptance_probability):
        self.N = N
        self.state = np.random.choice([0, 1], self.N).tolist()
        self.landscape = landscape
        self.fitness = self.landscape.query_fitness(self.state)
        self.state_cf = self.state
        self.fitness_cf = self.fitness
        self.kb = knowledge_base
        self.acceptance_probability = acceptance_probability
        
    def search(self, ): #usual human search
        next_state = list(self.state)
        next_state_cf = list(self.state_cf)
        next_index = self.kb[np.random.choice(len(self.kb))]
        #counterfactual is the same in that it tweaks the same index
        next_state[next_index] ^= 1 
        next_state_cf[next_index] ^= 1 
        
        if self.query_fitness(self.state) < self.query_fitness(next_state):
            self.state = next_state
            self.fitness = self.landscape.query_fitness(next_state)

        if self.query_fitness(self.state_cf) < self.query_fitness(next_state_cf):
            self.state_cf = next_state_cf
            self.fitness_cf = self.landscape.query_fitness(next_state_cf)
            
    def query_fitness(self, state):
         #Updated: Human calculates the fitness based on his knowledge base only
        return self.landscape.query_cog_fitness(state, self.kb)
            
    def review(self, aa_results):
        if self.query_fitness(self.state) < self.query_fitness(aa_results):
            if rand.uniform(0,1) < self.acceptance_probability: #even if aa results are better, human will only accept it with a certain probability
                self.state = [int(c) for c in aa_results]
                self.fitness = self.landscape.query_fitness(self.state)

#Artificial Agent - edit
class ArtificialAgent:
    
    def __init__(self, N, landscape, knowledge_base, computational_constraint, understand_capability):
        self.N = N
        self.state = np.random.choice([0, 1], self.N).tolist()
        self.landscape = landscape
        self.fitness = self.landscape.query_fitness(self.state)
        self.kb = knowledge_base
        self.con_p = computational_constraint #a number between 0 - 1 that defines the maximum number of bits that the AA can search: self.con * 2 ^ len(self.kb)
        self.con = int(computational_constraint * len(knowledge_base))
        self.cap = understand_capability #a number between 0 - 1, it represents how well the AA can understand the human dimensions
        
    def query_fitness(self, state, cond):
        #AA calculates fitness by looking at F|prompt
        return self.landscape.query_cog_fitness(state, cond)
        
    def search(self, state, prompt, human_kb):
        #human will give AA a state to start with, and prompt where to search
        #AA will then return the best solution it can find within computational constraints (such as time)
        curr_state = list(self.state)
        next_state = curr_state
        possibilities = []
        prompt_space = [int(i) for i in prompt if i or str(i) in self.kb] #prompt will limit the search space. Tells AA what to do
        
        for i in range(pow(2,len(prompt_space))): #get possibilities
            bit = bin(i)[2:]
            poss = list(state)
            if len(bit) < len(prompt_space):
                bit = "0"*(len(prompt_space)-len(bit)) + bit
            for j in range(len(bit)):
                poss[prompt_space[j]] = bit[j]
            possibilities.append(''.join(poss))
        
        possibilities.sort(key =  lambda x: hamming_dist(x, state))
        
        search_space = int(self.con_p*len(possibilities))
        bit = possibilities[0]
        cog_space = generate_ai_cognitive_space(self.kb, human_kb, self.cap)
        best_fitness = self.query_fitness(bit, cog_space) #calculates fitness based on own kb and also human's existing information
        best_bit = bit
        for i in range(1, search_space):
            bit = possibilities[i]
            fitness = self.query_fitness(bit, cog_space)
            if fitness > best_fitness:
                best_fitness = fitness
                best_bit = bit
        return best_bit

def baseline_model(N, K, aa_kb_idx, non_aa_kb_idx, computational_constraint, trials, prompt, understand_capabilities, overlap, acceptance_probability = 1):
    """
    Runs a baseline NK model with Human-AI Partnership
    N - Number of decision variables, forms NK landscape
    K - Number of dependencies, defines landscape ruggedness
    aa_kb_idx - Knowledge base for AI
    non_aa_kb_idx - index that is outside of aa_kb for human knowledge base to sample from
    computational_constraint - Defines the constraints under which the AA can search. Mathematically, it limits the number of points that AA can search in one search.
    trials - number of trials to run this model. Model will generate 'trials' number of landscapes and 'trial' * 10 number of agents for each landscape
    prompt - what the humans prompt the AA. Its a subset of aa_kb_idx, and is default across runs
    understand_capabilities - a list of understand capability
    acceptance probability: probability that human accepts the AA solutions, scaled to the fitness of the AA solution
    returns a dictionary of runs and their fitness values throughout the run
    """
    results_fitness = {}
    results_coord = {}
    counterfactual_fitness = {}
    counterfactual_coord = {}
    search_iteration = 100 #search 100 times
    run_cnt = 0
    for i in range(trials):
        np.random.seed(None)
        landscape = LandScape(N, K, None, None) #instantiate Landscape
        landscape.initialize(norm=True)
        for j in range(trials*10): #generate more agents
            #print(f'Landscape: {i}, Agent: {j}')
            #randomize understanding capability
            understand_capability = rand.choice(understand_capabilities)
            #randomize the human knowledge, but keep overlap and aa_kb_idx fixed
            human_kb_idx = rand.sample(aa_kb_idx, overlap)
            human_kb_idx += rand.sample(non_aa_kb_idx, rand.randint(0, len(non_aa_kb_idx) - 1))
            if len(human_kb_idx) == 0:
                human_kb_idx = [rand.choice(non_aa_kb_idx)]
            agent = Agent(N, landscape, human_kb_idx, acceptance_probability) #instantiate agent
            aa = ArtificialAgent(N, landscape, aa_kb_idx, computational_constraint, understand_capability) #instantiate AA
            fitness = [] #stores fitness over iterations
            fitness_cf = [] #stores counterfactual fitness
            coord = [] #stores coordinates/ states
            coord_cf = [] #stores coordinates/ states
            for iter in range(search_iteration):
                #first, human conduct search
                #print('initial:', agent.state, agent.fitness)
                agent.search()
                search_results = agent.state
                #then, human pass to AA to help
                aa_results = aa.search(''.join(str(num) for num in search_results), prompt, agent.kb)
                #human reviews the solutions given by AA, and chooses if he wants to take it in
                agent.review(aa_results)
                fitness.append(agent.landscape.query_fitness(agent.state))
                fitness_cf.append(agent.landscape.query_fitness(agent.state_cf))
                coord.append(str(agent.state))
                coord_cf.append(str(agent.state_cf))
                # print('searched:', agent.state, agent.fitness)
                # print('searched (counterfactual):', agent.state_cf, agent.fitness_cf)
            print(f'{datetime.datetime.now()}: run_{run_cnt})_landscape_{i}_humanknw_{len(human_kb_idx)}')
            results_fitness[f'run_{run_cnt}_landscape_{i}_humanknw_{len(human_kb_idx)}-exp'] = fitness
            results_fitness[f'run_{run_cnt}_landscape_{i}_humanknw_{len(human_kb_idx)}-cf'] = fitness_cf
            counterfactual_fitness['run_' + str(run_cnt)] = fitness_cf
            results_coord[f'run_{run_cnt}_landscape_{i}_humanknw_{len(human_kb_idx)}-exp'] = coord
            results_coord[f'run_{run_cnt}_landscape_{i}_humanknw_{len(human_kb_idx)}-cf'] = coord_cf
            counterfactual_coord['run_' + str(run_cnt)] = coord_cf
            run_cnt += 1
    return {"Experiment Fitness": results_fitness, "Experiment Bits": results_coord,
            "Counterfactual Fitness": counterfactual_fitness, "Counterfactual Bits": counterfactual_coord}

def job(N, K, aa_kb_idx, non_aa_kb_idx, computational_constraint, trials, prompt, understand_capabilities, overlap, current_directory):
    try:
        os.chdir(current_directory)
        worker_id = multiprocessing.current_process().name 
        print(f"Current Worker: {worker_id}")
        #code for multiprocessing
        path = f"../tmp/data_281024/comp_constraint_{computational_constraint}/overlap_{overlap}/k{K}/"
        os.makedirs(f"../tmp/data_281024/comp_constraint_{computational_constraint}/overlap_{overlap}/k{K}/", exist_ok=True)
        raw_data = baseline_model(N, K, aa_kb_idx, non_aa_kb_idx, computational_constraint, trials, prompt, understand_capabilities, overlap)
        fitness = pd.DataFrame(raw_data["Experiment Fitness"])
        fitness.to_csv(f'{path}exp_fit.csv')
        bits = pd.DataFrame(raw_data["Experiment Bits"])
        bits.to_csv(f'{path}exp_bit.csv')
        fitness_cf = pd.DataFrame(raw_data["Counterfactual Fitness"])
        fitness_cf.to_csv(f'{path}cf_fit.csv')
        bits_cf = pd.DataFrame(raw_data["Counterfactual Bits"])
        bits_cf.to_csv(f'{path}cf_bit.csv')
        #select specific timings to investigate
        #fitness.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 99]].to_csv(f'{path}exp_fit_condensed.csv')
        #bits.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 99]].to_csv(f'{path}exp_bit_condensed.csv')
        #fitness_cf.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 99]].to_csv(f'{path}cf_fit_condensed.csv')
        #bits_cf.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 99]].to_csv(f'{path}cf_bit_condensed.csv')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    #run model a few times
    N = 12
    k_lst = [11, 0]
    computational_constraints = [0.7, 1]
    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    aa_kb_idx = [6, 7, 8, 9, 10, 11]
    non_aa_kb_idx = [0, 1, 2, 3, 4, 5]
    trials = 100
    prompt = aa_kb_idx
    uhk = 2
    understand_capabilities = [0.2, 0.4, 0.6, 0.8, 0.9, 1]
    overlaps = [0, 1, 2, 3, 4, 5, 6]
    processes = []
    current_directory = os.getcwd()
    with multiprocessing.Pool() as pool:
        job_args = [
            (N, K, aa_kb_idx, non_aa_kb_idx, computational_constraint, trials, prompt, understand_capabilities, overlap, current_directory)
            for computational_constraint in computational_constraints
            for K in k_lst
            for overlap in overlaps
        ]
        pool.starmap(job, job_args)
        pool.close()
        pool.join()

# %%
