import numpy as np
import pickle
from evaluator import Evaluator
from revolve2.ci_group.logging import setup_logging
import pandas as pd
import re
from revolve2.examples.evaluate_single_robot import modified

data = pd.read_csv('/Users/corinnatriebold/Developer/revolve2/revolve2/Evo/Results/morphology_3.0_0.0/xbest/9_1_737497_xbest.csv')
agent = data.values.tolist()
# PARAMS = np.array([float(re.findall("\d+\.\d+", i)[0]) for i in agent[0]])
# print(PARAMS)
# PARAMS = np.array([-0.96630358,  0.99038008,  0.94955542, -0.00525027, -0.53647551,
#         0.00167203,  0.79198974, -0.0142178 ,  0.82295878,  0.99113653,
#        -0.7251427 , -0.86829821,  0.98791219])
PARAMS = np.array([0.2493, 0.1853, 0.625,  0.0073, 0.0546, 0.9455, 4.1057, 4.7721, 0.3898, 0.1103,
 0.2663, 2.8257, 1.3953])
def main(BODY) -> None:
    """Perform the rerun."""
    setup_logging()

    file = open('test', 'rb')
    cpg_network_structure = pickle.load(file)
    evaluator = Evaluator(
        headless=True,
        num_simulators=1,
        cpg_network_structure=cpg_network_structure,
        body=modified.select_morph(BODY)
    )
    fitness = evaluator.evaluate([PARAMS])
    print(fitness)
    return fitness[0]

if __name__ == "__main__":
    main([3, 0.0])
