import os
import pickle
import numpy as np
import pandas as pd
from revolve2.Evo.utils import evaluate2
from revolve2.examples.evaluate_single_robot import modified
import torch
from utils import generate_morphologies
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

torch.set_printoptions(precision=30)
file = open('brain', 'rb')
cpg_network_structure = pickle.load(file)

vars = [-1.5, -1.3, -0.9, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 0.9, 1.3, 1.5]
variations = generate_morphologies([1, 2, 3, 4, 5, 6], vars)
# [-1.4, -1, -0.9, -0.4, -0.2, 0.0, 0.2, 0.4, 0.9, 1, 1.4]
# [-1.5, -1.4, -1.3, -1.2,-1.1, -1, -0.9, -0.8, -0.7,-0.6, -0.5, -0.4,-0.3, -0.2, -0.1, 0.0, 0.1, 0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
# variations = generate_morphologies([1, 2, 3, 4, 5, 6], [-1.5, -1.4, -1.3, -1.2,-1.1, -1, -0.9, -0.8, -0.7,-0.6, -0.5, -0.4,-0.3, -0.2, -0.1, 0.0, 0.1, 0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

def comparison(agent, i):
    fitness = evaluate2(agent,
                        cpg_network_structure,
                        modified.select_morph(variations[i]))
    return fitness

path = 'Results_Generalists'
# path = 'Results_Testing'
# path = 'Results_Specialist'
type = 'xbest'
# type = 'generalist'
if __name__ == "__main__":
    files = os.listdir(f'{path}/{type}')
    overall = []
    for file in sorted(files):
        if file.endswith('pt'):
            print(file)
            agent = torch.load(f'{path}/{type}/{file}')
            compare = joblib.Parallel(n_jobs=1)(joblib.delayed(comparison)(agent, i)
                                                          for i in range(len(variations)))
            fitnesses = np.array(compare)
            sns.heatmap(np.array(fitnesses).reshape(6, len(vars)), vmin=0, vmax=4, cmap='rocket_r')
            plt.title(f'{file}')
            # plt.savefig(f'{path}/{file}.pdf')
            plt.show()
            fitness = evaluate2(agent, cpg_network_structure, modified.select_morph([0, 0.0]))
            print('Default', fitness)
            fitnesses = np.append(fitnesses, fitness)
            print(np.mean(fitnesses))
            overall.append(np.mean(fitnesses))

    # df = pd.DataFrame(overall)
    # df.to_csv('Testing_test.csv')