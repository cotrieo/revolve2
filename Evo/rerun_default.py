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
            agent = [ 0.257053375244140625000000000000, -0.126497730612754821777343750000,
         0.114709489047527313232421875000,  0.061258811503648757934570312500,
         0.060829147696495056152343750000,  0.045217394828796386718750000000,
         0.151055574417114257812500000000, -0.177375748753547668457031250000,
        -0.114454820752143859863281250000,  0.034653682261705398559570312500,
        -0.029643673449754714965820312500,  0.028568770736455917358398437500,
         0.148876756429672241210937500000]
            fitness = evaluate2(agent,
                                cpg_network_structure,
                                modified.select_morph([5, -1.5]))
            print(file, fitness)
            overall.append(fitness)

    sns.heatmap(np.array(overall).reshape(5, 6), vmin=0, vmax=4, cmap='rocket_r')
    # plt.savefig(f'{path}/{file}.pdf')
    plt.show()

    # df = pd.DataFrame(overall)
    # df.to_csv('Generalist_Default.csv')