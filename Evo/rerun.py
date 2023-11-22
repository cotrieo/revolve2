import pickle
from revolve2.Evo.utils import evaluate2
from revolve2.examples.evaluate_single_robot import modified
import torch

torch.set_printoptions(precision=30)
data = torch.load('/Users/corinnatriebold/Developer/revolve2/revolve2/Evo/Results_Generalist/generalist/0_1_885440_generalist.pt')
# data = torch.load('/Users/corinnatriebold/Downloads/0_1_656182_xbest.pt')
file = open('brain', 'rb')
cpg_network_structure = pickle.load(file)

if __name__ == "__main__":
    fitness = evaluate2(data, cpg_network_structure, modified.select_morph([1, 0.0]))
    print(fitness)