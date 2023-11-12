import os
import asyncio
import math
import torch
import numpy as np
from revolve2.ci_group import fitness_functions, terrains
from revolve2.ci_group.simulation import create_batch_single_robot_standard
from revolve2.modular_robot import ModularRobot, get_body_states_single_robot
from revolve2.modular_robot.brains import BrainCpgNetworkStatic
from revolve2.simulators.mujoco import LocalRunner


def generate_morphologies(parameter1_range, parameter2_range):
    morphologies = np.array(np.meshgrid(parameter1_range, parameter2_range)).T.reshape(-1, 2)
    return morphologies


def save_dataframe(dataframe, directory, filename):
    dataframe.to_csv(os.path.join(directory, filename), index=False)


def create_directories(path, subdirectories):
    for subdir in subdirectories:
        os.makedirs(os.path.join(path, subdir), exist_ok=True)


def save_dataframes(evals, best, generalist, generalist_evals, info, path):
    subdirectories = ['evals', 'generalist_evals']
    subs2 = ['xbest', 'generalist', ]
    create_directories(path, subdirectories)
    create_directories(path, subs2)

    file_names = [
        '{}_evals.csv'.format(info),
        '{}_generalist_evals.csv'.format(info)
    ]

    file_names2 = [
        '{}_xbest.pt'.format(info),
        '{}_generalist.pt'.format(info)
    ]
    dataframes = [evals, generalist_evals]
    tensors = [best, generalist]

    for tensor, subir, filename in zip(tensors, subs2, file_names2):
        torch.save(tensor, '{}/{}/{}'.format(path, subir, filename))

    for dataframe, subdir, filename in zip(dataframes, subdirectories, file_names):
        save_dataframe(dataframe, os.path.join(path, subdir), filename)


def evaluate2(agent, cpg_network_structure, body):
    brain = BrainCpgNetworkStatic.create_simple(
        params=np.array(agent),
        cpg_network_structure=cpg_network_structure,
        initial_state_uniform=math.pi / 2.0,
        dof_range_uniform=1.0,
    )

    robot = ModularRobot(body, brain)

    batch = create_batch_single_robot_standard(robot=robot, terrain=terrains.flat())
    runner = LocalRunner(headless=True)
    results = asyncio.run(runner.run_batch(batch))

    environment_results = results.environment_results[0]

    body_state_begin, body_state_end = get_body_states_single_robot(
        body, environment_results
    )

    xy_displacement = fitness_functions.xy_displacement(
        body_state_begin, body_state_end
    )
    return xy_displacement
