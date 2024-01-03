import os
import pickle
import numpy as np
import pandas as pd
from revolve2.Evo.utils import evaluate2
from revolve2.examples.evaluate_single_robot import modified
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

torch.set_printoptions(precision=30)
file = open('brain', 'rb')
cpg_network_structure = pickle.load(file)

vars = [-1.5, -1.3, -0.9, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 0.9, 1.3, 1.5]
motors = [1, 2, 3, 4, 5, 6]
morphologies = np.array(np.meshgrid(motors, vars)).T.reshape(-1, 2)

# [-1.4, -1, -0.9, -0.4, -0.2, 0.0, 0.2, 0.4, 0.9, 1, 1.4]
# [-1.5, -1.4, -1.3, -1.2,-1.1, -1, -0.9, -0.8, -0.7,-0.6, -0.5, -0.4,-0.3, -0.2, -0.1, 0.0, 0.1, 0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
# variations = generate_morphologies([1, 2, 3, 4, 5, 6], [-1.5, -1.4, -1.3, -1.2,-1.1, -1, -0.9, -0.8, -0.7,-0.6, -0.5, -0.4,-0.3, -0.2, -0.1, 0.0, 0.1, 0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

def comparison(agent, i):
    fitness = evaluate2(agent,
                        cpg_network_structure,
                        modified.select_morph(morphologies[i]), True)
    return fitness

# path = 'Results_Generalist'
path = 'Results_Testing'
# path = 'Results_Specialists'
# type = 'xbest'
type = 'generalist'
if __name__ == "__main__":
    files = os.listdir(f'{path}/{type}')
    overall = []
    splits = []
    defaults =[]
    print(files)
    for i in range(1, 3):
        for file in sorted(files):
            if file.startswith(f'{i}') and file.endswith('.pt'):
                print(file)
                agent = torch.load(f'{path}/{type}/{file}')
                compare = joblib.Parallel(n_jobs=4, require='sharedmem')(joblib.delayed(comparison)(agent, i)
                                                              for i in range(len(morphologies)))
                splits.append(np.array(compare))
                print(compare)
                fitness = evaluate2(agent, cpg_network_structure, modified.select_morph([0, 0.0]), True)
                print('Default', fitness)
                defaults.append(fitness)


        for k in range(len(splits)-1):
            res = np.max([splits[k], splits[k+1]], axis=0)
            splits[k+1] = res
            default = np.max([defaults[k], defaults[k+1]], axis=0)
            defaults[k + 1] = default

        fitnesses = np.array(splits[-1])
        sns.heatmap(np.array(fitnesses).reshape(6, len(vars)), vmin=-1, vmax=3, cmap='rocket_r')
        plt.title(f'{file}')
        plt.savefig(f'{path}/{file}.pdf')
        plt.show()
        fitnesses = np.append(fitnesses, defaults[-1])
        print(np.mean(fitnesses))
        overall.append(np.mean(fitnesses))


    df = pd.DataFrame(overall)
    df.to_csv(f'{path}.csv')