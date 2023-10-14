"""Rerun a robot with given body and parameters."""

import config
import numpy as np
import pandas as pd
import pickle
from evaluator import Evaluator
from revolve2.ci_group.logging import setup_logging
from revolve2.modular_robot.brains import (
    body_to_actor_and_cpg_network_structure_neighbour,
)
from revolve2.ci_group import modified
from revolve2.actor_controllers.cpg import CpgNetworkStructure, CpgPair
from revolve2.actor_controllers.cpg import Cpg
# These are set of parameters that we optimized using CMA-ES.
# You can copy your own parameters from the optimization output log.

# weird flip
PARAMS = np.array(
    [ 0.10209706, -0.93600196,  0.90297108, -0.04056276, -0.17203321,
        0.07839859, -0.92924995,  0.99066365, -0.97509642, -0.93872162,
        0.99987627, -0.08923148, -0.76304771]
)

# weird jump
# PARAMS = np.array([-0.12468656, -0.3842278 , -0.2279    , -0.0319296 ,  0.93032478,
#        -0.79062472,  0.33808627,  0.99540252,  0.97888288,  0.96248908,
#        -0.84870826, -0.69727372,  0.87909056])

# highest fitness = 5
# PARAMS = np.array(
#     [ 0.96267557,  0.19402816,  0.22640275, -0.98225126,  0.3024018 ,
#         0.12677748,  0.99642197,  0.00474394,  0.82592724,  0.95242033,
#        -0.81636629, -0.69288294,  0.92687656]
# )

# most common behaviour
# PARAMS = np.array(
#     [-0.01424386, -0.3877214 , -0.99828177,  0.2742341 ,  0.64417341,
#        -0.33797203,  0.71021989, -0.03378655,  0.89256247,  0.97801285,
#        -0.93766711, -0.92370213,  0.96884416]
# )

# weird flip: run 6
# PARAMS = np.array(
#     [ 0.9572968 , -0.45198869,  0.73781198, -0.80692572,  0.9098322 ,
#        -0.27272483,  0.45923107,  0.06397369,  0.47484437,  0.9616323 ,
#        -0.14927122, -0.9713818 ,  0.65165145]
# )

# testing
# PARAMS = np.array(
#     [-0.75671854,  0.51816266,  0.14979083, -0.59904403,  0.84826756,
#        -0.36094012,  0.97132693,  0.04788737,  0.999965  ,  0.90766866,
#        -0.99667139, -0.6092882 ,  0.11889928]
# )

def main(BODY) -> None:
    """Perform the rerun."""
    setup_logging()

    # _, cpg_network_structure = body_to_actor_and_cpg_network_structure_neighbour(
    #     # config.BODY
    #     modified.select_morph(BODY)
    # )

    # dbfile = open('test', 'ab')
    #
    # # source, destination
    # pickle.dump(cpg_network_structure, dbfile)
    # dbfile.close()
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


def generate_morphologies(parameter1_range, parameter2_range):
    morphologies = np.array(np.meshgrid(parameter1_range, parameter2_range)).T.reshape(-1, 2)
    return morphologies

servos = [0, 1, 2, 3, 4, 5, 6]
angles = [-1.5, -1.3, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.3, 1.5]

morphs = generate_morphologies(servos, angles)
fitnesses = []
bodies = []
if __name__ == "__main__":
    for morph in morphs:
        fitness = main(morph)
        print(morph, fitness)
        # fitnesses.append(fitness)
        # bodies.append(morph)
    # print(fitnesses)
    # df = pd.DataFrame({'morphology': bodies, 'fitness': fitnesses})
    # df.to_csv('feasible.csv')
