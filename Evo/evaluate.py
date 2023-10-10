import pickle
from revolve2.examples.robot_brain_cmaes.evaluator import Evaluator
from revolve2.ci_group.logging import setup_logging
from revolve2.examples.evaluate_single_robot import modified

def main(BODY, agent) -> None:
    setup_logging()

    file = open('test', 'rb')
    cpg_network_structure = pickle.load(file)
    evaluator = Evaluator(
        headless=True,
        num_simulators=1,
        cpg_network_structure=cpg_network_structure,
        body=modified.select_morph(BODY)
    )
    fitness = evaluator.evaluate([agent])
    return fitness[0]

if __name__ == "__main__":
    fitness = main(BODY, agent)