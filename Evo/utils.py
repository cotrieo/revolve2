import pandas as pd
import numpy as np
import os


def generate_morphologies(parameter1_range, parameter2_range):
    morphologies = np.array(np.meshgrid(parameter1_range, parameter2_range)).T.reshape(-1, 2)
    return morphologies

def save_dataframe(dataframe, directory, filename):
    dataframe.to_csv(os.path.join(directory, filename), index=False)


def create_directories(path, subdirectories):
    for subdir in subdirectories:
        os.makedirs(os.path.join(path, subdir), exist_ok=True)


def save_dataframes(evals, best, generalist, generalist_evals, info, path):
    subdirectories = ['xbest', 'generalist', 'evals', 'generalist_evals']

    create_directories(path, subdirectories)

    file_names = [
        '{}_evals.csv'.format(info),
        '{}_xbest.csv'.format(info),
        '{}_generalist.csv'.format(info),
        '{}_generalist_evals.csv'.format(info)
    ]

    dataframes = [evals, pd.DataFrame(best), pd.DataFrame(generalist), generalist_evals]

    for dataframe, subdir, filename in zip(dataframes, subdirectories, file_names):
        save_dataframe(dataframe, os.path.join(path, subdir), filename)
