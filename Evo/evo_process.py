import joblib
import asyncio
import random
import math
import torch
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from revolve2.ci_group import fitness_functions, terrains, modular_robots
from revolve2.ci_group.simulation import create_batch_single_robot_standard
from revolve2.modular_robot import ModularRobot, get_body_states_single_robot
from revolve2.modular_robot.brains import BrainCpgNetworkStatic
from revolve2.examples.evaluate_single_robot import modified
from revolve2.simulators.mujoco import LocalRunner
from evotorch import Problem
from evotorch.logging import StdOutLogger, PandasLogger
from evotorch.algorithms import XNES
from utils import save_dataframes, evaluate2
from evotorch import SolutionBatch


class Eval(Problem):
    def __init__(self, variations, cpg, counter):
        super().__init__(
            objective_sense="max",
            solution_length=13,
            initial_bounds=(-0.1, 0.1),
        )

        self.variations = variations
        self.env_counter = counter
        self.parameters = self.variations[self.env_counter]
        self.cpg_network_structure = cpg

    def evals(self, agent: torch.Tensor) -> float:

        brain = BrainCpgNetworkStatic.create_simple(
            params=np.array(agent),
            cpg_network_structure=self.cpg_network_structure,
            initial_state_uniform=math.pi / 2.0,
            dof_range_uniform=1.0,
        )
        body = modified.select_morph(self.parameters)
        robot = ModularRobot(body, brain)

        batch = create_batch_single_robot_standard(robot=robot, terrain=terrains.flat())
        runner = LocalRunner(headless=True)
        results = asyncio.run(runner.run_batch(batch))

        environment_results = results.environment_results[0]

        body_state_begin, body_state_end = get_body_states_single_robot(
            body, environment_results
        )

        xy_displacement = fitness_functions.xy_displacement(
            body_state_begin, body_state_end)

        return xy_displacement

    def _evaluate_batch(self, solutions: SolutionBatch):
        solutions.set_evals(
            torch.FloatTensor(joblib.Parallel(n_jobs=6)(joblib.delayed(self.evals)(i) for i in solutions.values)))
        if len(self.variations) > 1:
            self.env_counter += 1
        else:
            self.env_counter = 0

        if self.env_counter >= len(self.variations):
            self.env_counter = 0

        self.parameters = self.variations[self.env_counter]

    def comparison(self, agent, i):

        fitness = evaluate2(agent,
                            self.cpg_network_structure,
                            modified.select_morph(self.variations[i]))
        return fitness

    def split(self, good_fitness_scores, generalist_avg_fit, generalist_dev, generalist_weights):
        break_stat = False
        good_envs = []
        bad_envs = []

        for i in range(len(self.variations)):
            if good_fitness_scores[i] < (generalist_avg_fit + generalist_dev):
                good_envs.append(self.variations[i])
            else:
                # add underperformed variations to bin
                bad_envs.append(list(self.variations[i]))

        if len(good_envs) == 0:
            print('No more envs')
            break_stat = True
        elif len(good_envs) == len(self.variations):
            print('No more bad envs')
            break_stat = True

        # replace set of variations with only the good variations and re-check their fitness
        self.variations = np.array(good_envs)

        compare_after = joblib.Parallel(n_jobs=4)(joblib.delayed(self.comparison)
                                                  (generalist_weights, i)
                                                  for i in range(len(self.variations)))

        generalist_scores = np.array(compare_after)
        new_avg_fit = np.mean(generalist_scores)

        self.env_counter = 0

        return break_stat, bad_envs, self.variations, generalist_scores, new_avg_fit


class Algo:
    def __init__(self, path, variations, config, run_id, cluster_id, generation):
        self.variations = variations
        self.path = path
        self.max_eval = generation
        self.cluster_id = cluster_id
        self.run_id = run_id
        self.max_fitness = config["maxFitness"]
        self.initial_stdev = config['stdev_init']
        self.initial_bounds = config["initial_bounds"]
        self.actors = config['actors']
        self.seed = random.randint(0, 1000000)
        file = open('brain', 'rb')
        self.cpg_network_structure = pickle.load(file)

    def comparison(self, agent, i):
        fitness = evaluate2(agent,
                            self.cpg_network_structure,
                            modified.select_morph(self.variations[i]))
        return fitness

    # main function to run the evolution
    def main(self):

        problem = Eval(self.variations, self.cpg_network_structure, 0)
        searcher = XNES(problem, stdev_init=self.initial_stdev)

        improved = 0
        generalist_std = 0
        prev_pop_best_fitness = 0
        generalist_weights = 0
        generation = 0
        current_pop_best_fitness = -self.max_eval
        generalist_avg_fit = -self.max_eval
        generalist_scores = np.zeros(len(self.variations))
        good_fitness_scores = np.zeros(len(self.variations))
        number_environments = []
        bad_environments = []
        generalist_avg_history = []
        general_std_history = []
        general_min_fitness_history = []
        general_max_fitness_history = []

        pandas_logger = PandasLogger(searcher)
        print('Number of Environments: ', len(self.variations))
        logger = StdOutLogger(searcher, interval=1)
        torch.set_printoptions(precision=30)

        while generation < self.max_eval:

            # take one step of the evolution and identify the best individual of a generation
            searcher.step()
            index_best = searcher.population.argbest()
            xbest_weights = searcher.population[index_best].values

            # if current best fitness is smaller than new best fitness replace the current fitness and xbest
            if current_pop_best_fitness < searcher.status.get('best_eval'):
                current_pop_best_fitness = searcher.status.get('best_eval')
                improved = searcher.status.get('iter')
                xbest = xbest_weights.detach().clone()

            # if we are running more than 1 variation
            if len(self.variations) > 1:

                # test xbest on all individuals in the morphology set
                compare = joblib.Parallel(n_jobs=self.actors)(joblib.delayed(self.comparison)(xbest, i)
                                                              for i in range(len(generalist_scores)))

                generalist_scores = np.array(compare)

                # check the average fitness score of the morphologies
                new_avg_fit = np.mean(generalist_scores)

                # log the info about the evolution of the generalist
                generalist_avg_history.append(new_avg_fit)
                generalist_new_std = np.std(generalist_scores)
                general_std_history.append(generalist_new_std)
                general_min_fitness_history.append(np.min(generalist_scores))
                general_max_fitness_history.append(np.max(generalist_scores))

                # if current generalist has a smaller avg score than new generalist replace avg score and weights
                if generalist_avg_fit < new_avg_fit:
                    generalist_avg_fit = new_avg_fit
                    generalist_std = generalist_new_std

                    print('Generalist score: ', generalist_avg_fit)

                    good_fitness_scores = generalist_scores.copy()
                    generalist_weights = xbest_weights.detach().clone()

                    # REMOVE LATER - ONLY FOR TESTING
                    if len(self.variations) == 25:
                        sns.heatmap(generalist_scores.reshape((5, 5)), vmin=-5, vmax=5, annot=True)
                        plt.show()

                # check if evolution has stagnated
                if (searcher.status.get('iter') - improved) % int(np.ceil(self.max_eval * 0.05)) == 0:

                    if current_pop_best_fitness != prev_pop_best_fitness:
                        prev_pop_best_fitness = current_pop_best_fitness
                    else:
                        # if the evolution has stagnated check the generalist fitness scores
                        break_stat, bad_envs, self.variations, generalist_scores, new_avg_fit = problem.split(
                            good_fitness_scores,
                            generalist_avg_fit,
                            generalist_std,
                            generalist_weights)

                        if new_avg_fit < generalist_avg_fit:
                            good_fitness_scores = generalist_scores.copy()

                        if len(bad_envs) > 0:
                            for env in bad_envs:
                                bad_environments.append(env)
                            print(bad_environments)

                        if break_stat == True:
                            break

                        improved = searcher.status.get('iter')
                        print(' no_envs : ', len(self.variations))

            # if there is only one morphology generalist = xbest
            elif len(self.variations) == 1:
                generalist_avg_fit = current_pop_best_fitness
                generalist_weights = xbest

            # track the number of envs
            number_environments.append(len(self.variations))
            generation = searcher.status.get('iter')

            # if desired fitness is found terminate evolution
            if generalist_avg_fit > self.max_fitness:
                print('Found best')
                break

        # data logging
        evals = pandas_logger.to_dataframe()

        if len(number_environments) != len(evals):
            number_environments.append(len(self.variations))

        evals['no_envs'] = number_environments

        generalist_evals = pd.DataFrame(
            {'Mean': generalist_avg_history, 'STD': general_std_history,
             'Best': general_min_fitness_history, 'Worst': general_max_fitness_history})

        info = '{}_{}_{}'.format(self.run_id, self.cluster_id, self.seed)

        save_dataframes(evals, xbest, generalist_weights, generalist_evals, info, self.path)

        plt.plot(evals['best_eval'])
        plt.ylabel('Fitness')
        plt.xlabel('Generations')
        plt.title('{}'.format(info))
        plt.savefig('{}/{}.pdf'.format(self.path, info))
        plt.show()

        return generation, np.array(bad_environments)
