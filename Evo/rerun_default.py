import os
import pickle
import numpy as np
from revolve2.Evo.utils import evaluate2
from revolve2.examples.evaluate_single_robot import modified
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

torch.set_printoptions(precision=30)
file = open('brain', 'rb')
cpg_network_structure = pickle.load(file)
path = 'Results_Generalist'
# path = 'Results_Specialist'
type = 'generalist'
# type = 'xbest'

if __name__ == "__main__":
    files = os.listdir(f'{path}/{type}')
    overall = []
    for file in files:
        if file.endswith('pt'):
            torch.set_printoptions(precision=30)

            agent = torch.load(f'{path}/{type}/{file}')

            fitness = evaluate2(agent,
                                cpg_network_structure,
                                modified.select_morph([0, 0.0]), True)
            print(file, fitness)
            overall.append(fitness)

    sns.heatmap(np.array(overall).reshape(3, 7), vmin=0, vmax=2, cmap='rocket_r')
    plt.savefig(f'{path}/{file}.pdf')
    plt.show()

    df = pd.DataFrame(overall)
    df.to_csv(f'{path}_Default.csv')