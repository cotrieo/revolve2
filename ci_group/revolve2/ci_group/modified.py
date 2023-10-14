import numpy as np
import random
from revolve2.modular_robot import ActiveHinge, Body, Brick, InActiveHingeLower, InActiveHinge, RightAngles
import itertools
import random
import pickle


def select_morph(morph):
    if morph[0] == 1:
        body = gecko_mod_front_left(morph[1])
    elif morph[0] == 2:
        body = gecko_mod_front_right(morph[1])
    elif morph[0] == 3:
        body = gecko_mod_core_back(morph[1])
    elif morph[0] == 4:
        body = gecko_mod_back(morph[1])
    elif morph[0] == 5:
        body = gecko_mod_back_left(morph[1])
    elif morph[0] == 6:
        body = gecko_mod_back_right(morph[1])
    else:
        body = gecko_mod()

    return body

def gecko_mod() -> Body:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)

    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)

    body.finalize()
    return body

def gecko_mod_front_left(degree) -> Body:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = InActiveHinge(degree, 0.0)
    body.core.left.front = InActiveHingeLower(0.0)
    body.core.left.front.front = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)

    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)

    body.finalize()
    return body


def gecko_mod_front_right(degree) -> Body:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = InActiveHinge(degree, 0.0)
    body.core.right.front = InActiveHingeLower(0.0)
    body.core.right.front.front = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)

    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)

    body.finalize()
    return body


def gecko_mod_core_back(degree) -> Body:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()
    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)

    body.core.back = InActiveHinge(degree, np.pi / 2.0)
    body.core.back.front = InActiveHingeLower(0.0)
    body.core.back.front.front = Brick(-np.pi / 2.0)


    body.core.back.front.front.front = ActiveHinge(np.pi / 2.0)
    body.core.back.front.front.front.attachment = Brick(-np.pi / 2.0)

    body.core.back.front.front.front.attachment.left = ActiveHinge(0.0)
    body.core.back.front.front.front.attachment.left.attachment = Brick(0.0)

    body.core.back.front.front.front.attachment.right = ActiveHinge(0.0)
    body.core.back.front.front.front.attachment.right.attachment = Brick(0.0)

    body.finalize()
    return body


def gecko_mod_back(degree) -> Body:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front = InActiveHinge(degree, np.pi / 2.0)
    body.core.back.attachment.front.front = InActiveHingeLower(0.0)
    body.core.back.attachment.front.front.front = Brick(-np.pi / 2.0)

    body.core.back.attachment.front.front.front.left = ActiveHinge(0.0)
    body.core.back.attachment.front.front.front.left.attachment = Brick(0.0)

    body.core.back.attachment.front.front.front.right = ActiveHinge(0.0)
    body.core.back.attachment.front.front.front.right.attachment = Brick(0.0)

    body.finalize()
    return body


def gecko_mod_back_left(degree) -> Body:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front.attachment.left = InActiveHinge(degree, 0.0)
    body.core.back.attachment.front.attachment.left.front = InActiveHingeLower(0.0)
    body.core.back.attachment.front.attachment.left.front.front = Brick(0.0)

    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)

    body.finalize()
    return body


def gecko_mod_back_right(degree) -> Body:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)

    body.core.back.attachment.front.attachment.right = InActiveHinge(degree, 0.0)
    body.core.back.attachment.front.attachment.right.front = InActiveHingeLower(0.0)
    body.core.back.attachment.front.attachment.right.front.front = Brick(0.0)

    body.finalize()
    return body
