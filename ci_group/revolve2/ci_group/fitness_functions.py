"""Standard fitness functions for modular robots."""

import math

from revolve2.modular_robot import BodyState

from typing import Callable, List, Optional, Union

import torch
from torch import nn
import numpy as np
from pyrr import Vector3, Quaternion

from revolve2.simulation.running._results import EnvironmentState

def xy_displacement(begin_state: BodyState, end_state: BodyState) -> float:
    """
    Calculate the distance traveled on the xy-plane by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    # if end_state.core_orientation[0] < 0:
    #     penalty = -2
    # else:
    #     penalty = 0

    # return math.sqrt(
    #     (begin_state.core_position[0] - end_state.core_position[0]) ** 2  # x-position
        # + ((begin_state.core_position[1] - end_state.core_position[1]) ** 2) # y-position

    # )
    # print((begin_state.core_position[1] - end_state.core_position[1]) ** 2)

    # x_mov = begin_state.core_position[0] - end_state.core_position[0]

    x_mov = end_state.core_position[0] - begin_state.core_position[0]

    return x_mov

def get_fitness(states: List[EnvironmentState]) -> float:
    """
    Fitness function.
    The fitness is the distance traveled minus the sum of squared actions (to penalize large movements)"""
    print(states)
    print(states[0].actor_states[0])
    actions = 0
    asymmetry = 0
    num_limbs = len(states[0].actor_states[0].dof_state)
    print(states)
    print(states[0].actor_states[0])
    for i in range(1, len(states), 2):

        action = (np.square(states[i - 1].actor_states[0].dof_state - states[i].actor_states[0].dof_state).sum()) / num_limbs

        asymmetry += np.std(np.square(states[i - 1].actor_states[0].dof_state)) + np.std(np.square(states[i].actor_states[0].dof_state))

        if action == 0:
            action = 0.5  # Penalize no movement

        actions += action

    distance = 3 * ((states[0].actor_states[0].position[0] - states[-1].actor_states[0].position[0]) ** 2)
               # + ((states[0].actor_states[0].position[1] - states[-1].actor_states[0].position[1]) ** 2)

    return distance - actions