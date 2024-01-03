import json
import os
from utils import generate_morphologies
from evo_process import Algo


def experiment_run(config):
    runs = config['runs']
    parameter1_range, parameter2_range = config['parameter1'], config['parameter2']

    for i in range(runs):
        variations = generate_morphologies(parameter1_range, parameter2_range)
        cluster_count = 0
        generations = config['generations']
        folder_name = config['filename']

        while len(variations) != 0:
            if generations > 1:
                cluster_count += 1
                i += 4
                path = f"{folder_name}/"
                os.makedirs(path, exist_ok=True)
                run = Algo(path=path, variations=variations,
                           config=config, generation=generations, run_id=i, cluster_id=cluster_count)
                generation, variations = run.main()
                generations = generations - generation
            else:
                break

if __name__ == "__main__":
    config = json.load(open('gecko.json'))
    experiment_run(config)
