"""
Set up an experiment that optimizes the brain of a given robot body using CMA-ES.

As the body is static, the genotype of the brain will be a fixed length real-valued vector.

Before starting this tutorial, it is useful to look at the 'experiment_setup' and 'evaluate_multiple_isolated_robots' examples.
It is also nice to understand the concept of a cpg brain, although not really needed.

You learn:
- How to optimize the brain of a robot using CMA-ES.
"""

import logging
import numpy as np
import cma
import pickle
import config
import pandas as pd
import matplotlib.pyplot as plt
from evaluator import Evaluator
from revolve2.ci_group.logging import setup_logging
from revolve2.ci_group.rng import seed_from_time
from revolve2.modular_robot.brains import (
    body_to_actor_and_cpg_network_structure_neighbour,
)
from pathlib import Path
from tqdm import tqdm

from revolve2.ci_group import modified


def run(BODY) -> None:
    # Empty Results folder
    [f.unlink() for f in Path("./Results_default").glob("*") if f.is_file()]

    setup_logging(file_name=f"./Results_default/log_{BODY}.txt")
    file = open('test', 'rb')
    cpg_network_structure = pickle.load(file)

    for i in tqdm(range(config.NUM_RUNS)):
        """Run the experiment."""
        fbest = []
        max_fitness = 0
        early_stopping_tolerance = 50
        generations_since_improvement = 0

        # Get the actor and cpg network structure for the body of choice.
        # The cpg network structure describes the connections between neurons in the cpg brain.
        # _, cpg_network_structure = body_to_actor_and_cpg_network_structure_neighbour(
        #     modified.select_morph(BODY)
        # )

        # Intialize the evaluator that will be used to evaluate robots.
        evaluator = Evaluator(
            headless=True,
            num_simulators=config.NUM_SIMULATORS,
            cpg_network_structure=cpg_network_structure,
            body=modified.select_morph(BODY),
        )

        # Initial parameter values for the brain.
        initial_mean = cpg_network_structure.num_connections * [0.5]

        # We use the CMA-ES optimizer from the cma python package.
        # Initialize the cma optimizer.
        options = cma.CMAOptions()
        options.set("bounds", [-1.0, 1.0])
        # The cma package uses its own internal rng.
        # Instead of creating our own numpy rng, we use our seed to initialize cma.
        rng_seed = seed_from_time() % 2**32  # Cma seed must be smaller than 2**32.
        options.set("seed", rng_seed)
        opt = cma.CMAEvolutionStrategy(initial_mean, config.INITIAL_STD, options)

        generation_index = 0

        # Run cma for the defined number of generations.
        logging.info("Start optimization process.")
        while generation_index < config.NUM_GENERATIONS:
            logging.info(f"Generation {generation_index + 1} / {config.NUM_GENERATIONS}.")

            # Get the sampled solutions(parameters) from cma.
            solutions = opt.ask()

            # Evaluate them. Invert because fitness maximizes, but cma minimizes.
            fitnesses = -evaluator.evaluate(solutions)

            # Tell cma the fitnesses.
            opt.tell(solutions, fitnesses)

            logging.info(f"{opt.result.xbest=} {opt.result.fbest=}")
            fbest.append(opt.result.fbest)
            # Increase the generation index counter.
            generation_index += 1

            current_best = opt.result.fbest * -1
            if current_best > max_fitness:
                max_fitness = current_best
                generations_since_improvement = 0
            else:
                generations_since_improvement += 1
                if generations_since_improvement > early_stopping_tolerance:
                    logging.info(f"Convergence -> Early stopping...")
                    break

        logging.info(f"End of run: {i}")

        # Save fitness
        df = pd.DataFrame({'fbest':fbest})
        df['fbest'] = df['fbest'] * -1
        df.to_csv(f'./Results_default/log_{i}_{BODY}.csv', index=False)

        # Plot fitnes
        plt.figure(figsize=(12,8))
        df['fbest'].plot()
        plt.title('Morphology: {}'.format(BODY))
        plt.xlabel('Generation')
        plt.ylabel('Best fitness')
        plt.savefig(f'./Results_default/fig_{i}_{BODY}.pdf')
        # plt.show()


def generate_morphologies(parameter1_range, parameter2_range):
    morphologies = np.array(np.meshgrid(parameter1_range, parameter2_range)).T.reshape(-1, 2)
    return morphologies

servos = [1, 2, 3, 4, 5, 6]
# angles = [-1.5, -1.3, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.3, 1.5]
angles = [-1.5, -0.5, 0.0, 0.5, 1.5]
morphs = generate_morphologies(servos, angles)
morphs = np.append(morphs, [0, 0.0])
morphs = morphs.reshape((31,2))

fitnesses = []
bodies = []
if __name__ == "__main__":
    # for morph in morphs:
    run([0, 0.0])
