"""
Simulate a single modular robot, and then calculate its xy displacement.

To understand simulation, first see the 'simulate_single_robot' example.

You learn:
- How to process simulation results.
"""

import asyncio
import pickle
from revolve2.ci_group import fitness_functions, terrains, modular_robots
from revolve2.examples.evaluate_single_robot import modified
from revolve2.ci_group.logging import setup_logging
from revolve2.ci_group.rng import make_rng
from revolve2.ci_group.simulation import create_batch_single_robot_standard
from revolve2.modular_robot import ModularRobot, get_body_states_single_robot
from revolve2.modular_robot.brains import BrainCpgNetworkNeighborRandom, BrainCpgNetworkNeighbor, BrainCpgNetworkStatic
from revolve2.simulators.mujoco import LocalRunner
import random
import numpy as np

def main(morph) -> None:
    """Run the simulation."""
    # Set up standard logging.
    setup_logging()

    # Set up a random number generater.
    RNG_SEED = 5
    rng = make_rng(RNG_SEED)

    # Create the robot.
    # body = modular_robots.gecko()
    body = modified.gecko_mod(morph)
    brain = BrainCpgNetworkNeighborRandom(rng)
    # brain = BrainCpgNetworkStatic(par)
    robot = ModularRobot(body, brain)


    # Create the simulation batch.
    batch = create_batch_single_robot_standard(robot=robot, terrain=terrains.flat())

    # Create the runner.
    # We set the headless parameters, which will run the simulation as fast as possible.
    # runner = LocalRunner(headless=True)
    runner = LocalRunner()
    # Running the batch returns simulation results.
    results = asyncio.run(runner.run_batch(batch))

    # Get the results of the first environment, which is the only one since we set up a single simulation environment.
    environment_results = results.environment_results[0]

    # We have to map the simulation results back to robot body space.
    # This function calculates the state of the robot body at the start and end of the simulation.
    body_state_begin, body_state_end = get_body_states_single_robot(
        body, environment_results
    )

    # Calculate the xy displacement from the body states.
    xy_displacement = fitness_functions.xy_displacement(
        body_state_begin, body_state_end
    )

    print(xy_displacement)

pickle_in = open("morphologies.pickle","rb")
morphologies = pickle.load(pickle_in)

def generate_morphologies(parameter1_range, parameter2_range):
    # parameter1_values = np.arange(parameter1_range[0], parameter1_range[1], step_sizes[0])
    # parameter2_values = np.arange(parameter2_range[0], parameter2_range[1], step_sizes[1])

    morphologies = np.array(np.meshgrid(parameter1_range, parameter2_range)).T.reshape(-1, 2)
    print(len(morphologies))
    return morphologies


servos = [1, 2, 3, 4, 5, 6]
angles = [-1.5, -1.3, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.3, 1.5]

morphs = generate_morphologies(servos, angles)

# if __name__ == "__main__":
for morph in morphs:
    print(morph)
    main(morph)
