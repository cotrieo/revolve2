"""Standard fitness functions for modular robots."""

import math

from revolve2.modular_robot import BodyState


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
    return math.sqrt(
        (begin_state.core_position[0] - end_state.core_position[0]) ** 2  # x-position
        + ((begin_state.core_position[1] - end_state.core_position[1]) ** 2) # y-position

    )