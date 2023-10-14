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
from revolve2.examples.evaluate_single_robot import modified
from revolve2.actor_controllers.cpg import CpgNetworkStructure, CpgPair
from revolve2.actor_controllers.cpg import Cpg
# These are set of parameters that we optimized using CMA-ES.
# You can copy your own parameters from the optimization output log.
# PARAMS = np.array(
#     [
#         0.96349864,
#         0.71928482,
#         0.97834176,
#         0.90804766,
#         0.69150098,
#         0.48491278,
#         0.40755897,
#         0.99818664,
#         0.9804162,
#         -0.34097883,
#         -0.01808513,
#         0.76003573,
#         0.66221044,
#     ]
# )

PARAMS = np.array([-0.96630358,  0.99038008,  0.94955542, -0.00525027, -0.53647551,
        0.00167203,  0.79198974, -0.0142178 ,  0.82295878,  0.99113653,
       -0.7251427 , -0.86829821,  0.98791219])
# PARAMS = np.array([-0.44375534, 0.99499762, -0.60628628, 0.80725963, -0.91015117,
#                    0.97603211, -0.99989902, 0.53095133, 0.99875929, -0.92524671, 0.0, 0.0, 0.0])

# PARAMS = np.array([ 0.89360297,  0.17863396,  0.30116421,  0.37762144, -0.18592681,
#         0.79980659])

# PARAMS = np.array([ 0.92500359,  0.97976839,  0.38624175,  0.52012928, -0.31824363,
#                     0.79952226, -0.37602673,  0.13259782,  0.89573339])
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
    main([0, 0.0])
    # for morph in morphs:
    #     fitness = main(morph)
    #     print(morph, fitness)
        # fitnesses.append(fitness)
        # bodies.append(morph)
    # print(fitnesses)
    # df = pd.DataFrame({'morphology': bodies, 'fitness': fitnesses})
    # df.to_csv('feasible.csv')
