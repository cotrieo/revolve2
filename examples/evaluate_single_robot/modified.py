import numpy as np
import random
from revolve2.modular_robot import ActiveHinge, Body, Brick, InActiveHingeLower, InActiveHinge, RightAngles

import itertools
import random
import pickle

def gecko_mod(angles) -> Body:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()
    angle_core_left = angles[0]
    angle_core_right = angles[1]
    body.core.left = ActiveHinge((angle_core_left * (np.pi / 180)), 0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge((angle_core_right * (np.pi / 180)),  0.0)
    body.core.right.attachment = Brick(0.0)

    angle_body_core = angles[2]
    body.core.back = ActiveHinge((angle_body_core * (np.pi / 180)), np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    angle_body_core_back = angles[3]
    body.core.back.attachment.front = ActiveHinge((angle_body_core_back * (np.pi / 180)),  np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)
    angle_back_left = angles[4]
    # body.core.back.attachment.front.attachment.left = ActiveHinge((angle_back_left * (np.pi / 180)), 0.0)
    body.core.back.attachment.front.attachment.left = InActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.front = InActiveHingeLower(0.0)
    body.core.back.attachment.front.attachment.left.front.front = Brick(0.0)
    angle_back_right = angles[5]
    body.core.back.attachment.front.attachment.right = ActiveHinge((angle_back_right * (np.pi / 180)), 0.0)
    body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)

    body.finalize()
    return body


def make_morphologies(angles):


    all_combinations = list(itertools.combinations_with_replacement(angles, 6))
    good_combinations = []


    for combi in all_combinations:
        count = list(combi).count(1)
        if count <= 3:
            good_combinations.append(combi)


    morph_array = np.array(sorted(good_combinations))

    # write to file
    pickle_out = open("morphologies.pickle","wb")
    pickle.dump(morph_array, pickle_out)
    pickle_out.close()

    # check to confirm
    pickle_in = open("morphologies.pickle","rb")
    morphologies = pickle.load(pickle_in)
    print(morphologies)

# use this to create more morphologies
angles = [1, 30, 45, 60, 90]
# make_morphologies(angles)

# set all the servos to a fixed degree, choose one random servo and change that
# too many variations right now,
# see the result even if there is a large jump (most interesting)
# generate graph showing similarities
# malfunction in how angle is stuck
# distance in graph,
# measure: how many malfunctions of the servos