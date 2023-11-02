import pickle
import numpy as np
import pandas as pd
import numpy as np
import re
from revolve2.examples.robot_brain_cmaes.evaluator import Evaluator
from revolve2.ci_group.logging import setup_logging
from revolve2.examples.evaluate_single_robot import modified

# data = pd.read_csv('Results/generalist/0_1_616036_generalist.csv')
data = pd.read_csv('Results/generalist/0_2_78766_generalist.csv')
agent = data.values.tolist()
PARAMS = np.array([float(re.findall("\d+\.\d+", i)[0]) for i in agent[0]])
print(PARAMS)

def main(BODY) -> None:
    """Perform the rerun."""
    setup_logging()

    file = open('test', 'rb')
    cpg_network_structure = pickle.load(file)
    evaluator = Evaluator(
        headless=False,
        num_simulators=1,
        cpg_network_structure=cpg_network_structure,
        body=modified.select_morph(BODY)
        # body=config.BODY,
    )
    fitness = evaluator.evaluate([PARAMS])
    return fitness[0]


if __name__ == "__main__":
    fitness = main([0, 0.0])
    print(fitness)
