import json
from utils import generate_morphologies
from evo_process import Algo
import os
import sys


def experiment_run(config):
    runs = config['runs']
    parameter1_range, parameter2_range = config['parameter1'], config['parameter2']
    variations = generate_morphologies(parameter1_range, parameter2_range)

    for i in range(runs):
        cluster_count = 0
        generations = config['generations']
        folder_name = config['filename']

        while len(variations) != 0:
            cluster_count += 1
            path = f"{folder_name}/"
            os.makedirs(path, exist_ok=True)
            run = Algo(path=path, variations=variations,
                       config=config, generation=generations, run_id=i, cluster_id=cluster_count)
            generation, variations = run.main()
            generations = generations - generation


if __name__ == "__main__":
    config = json.load(open('gecko.json'))
    experiment_run(config)
