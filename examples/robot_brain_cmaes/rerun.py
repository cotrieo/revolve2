"""Rerun a robot with given body and parameters."""

import config
import numpy as np
from evaluator import Evaluator
from revolve2.ci_group.logging import setup_logging
from revolve2.modular_robot.brains import (
    body_to_actor_and_cpg_network_structure_neighbour,
)

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

# PARAMS = np.array([ 0.9772014 , -0.46552178, -0.77052453,  0.94828452, -0.66410415,
#         0.94865957,  0.92837354,  0.99174603,  0.94921237, -0.99728289])

# PARAMS = np.array([ 0.43256757, -0.08451429,  0.43358328,  0.99255331, -0.99928277,
#         0.99948187, -0.99241477,  0.89694624,  0.99917603, -0.99985895])  # right back 30 degree down value = 0.5

# PARAMS = np.array([ 0.99640073,  0.13683136, -0.26971396,  0.55017667,  0.97049723,
#        -0.98738748,  0.76562931, -0.47012102, -0.99898643,  0.98406012]) # right back 0 degree blocked = value = 0.0

PARAMS = np.array([-0.44375534,  0.99499762, -0.60628628,  0.80725963, -0.91015117,
        0.97603211, -0.99989902,  0.53095133,  0.99875929, -0.92524671])
def main() -> None:
    """Perform the rerun."""
    setup_logging()

    _, cpg_network_structure = body_to_actor_and_cpg_network_structure_neighbour(
        config.BODY
    )

    evaluator = Evaluator(
        headless=False,
        num_simulators=1,
        cpg_network_structure=cpg_network_structure,
        body=config.BODY,
    )
    evaluator.evaluate([PARAMS])


if __name__ == "__main__":
    main()
