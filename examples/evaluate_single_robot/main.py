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


if __name__ == "__main__":
    ind = random.randint(0, len(morphologies)-1)
    morph = list([60,60,60,60,60,1])
    print('Morphology no: {}'.format(ind))
    print('angle_core_left: {},\nangle_core_right: {},\nangle_body_core: {},\nangle_body_core_back: {},\nangle_back_left: {},\nangle_back_right: {}'.format(*morph))
    main(morph)
