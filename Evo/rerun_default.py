import os
import pickle
import numpy as np
from revolve2.Evo.utils import evaluate2
from revolve2.examples.evaluate_single_robot import modified
import torch
import seaborn as sns
import matplotlib.pyplot as plt

torch.set_printoptions(precision=30)
file = open('brain', 'rb')
cpg_network_structure = pickle.load(file)
# path = 'Results_Generalists'
path = 'Results/Results_Specialist'
type = 'generalist'
# type = 'xbest'

if __name__ == "__main__":
    files = os.listdir(f'{path}/{type}')
    overall = []
    for file in files:
        if file.endswith('pt'):
            torch.set_printoptions(precision=30)

            # agent = torch.load(f'{path}/{type}/{file}')
            agent = [-0.321430474519729614257812500000,  0.187029451131820678710937500000,
         0.040301993489265441894531250000, -0.312059044837951660156250000000,
         0.025743119418621063232421875000, -0.266363412141799926757812500000,
         0.561942338943481445312500000000, -0.047860540449619293212890625000,
         0.417027026414871215820312500000,  0.299402832984924316406250000000,
        -0.576709151268005371093750000000, -0.543886661529541015625000000000,
         0.516893386840820312500000000000]
            fitness = evaluate2(agent,
                                cpg_network_structure,
                                modified.select_morph([0, 0.0]))
            print(file, fitness)
            overall.append(fitness)

    sns.heatmap(np.array(overall).reshape(5, 6), vmin=0, vmax=4, cmap='rocket_r')
    # plt.savefig(f'{path}/{file}.pdf')
    plt.show()

    # df = pd.DataFrame(overall)
    # df.to_csv('Generalist_Default.csv')