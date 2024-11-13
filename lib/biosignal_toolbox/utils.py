# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from yaml import safe_load
from datetime import datetime

from sys import exit
import warnings
warnings.filterwarnings('ignore')

# *********************************************************************************
# ************************* Functions *********************************************
# *********************************************************************************

def loadConfig(filename=''):
    """
    This function loads the yaml configuration file into a dictionary.

    Parameters
    ----------
    filename : str
        name of the config file
    """
    try:
        config_dir = getRootPath() / 'config'
        with open(config_dir / filename, 'r') as file:
            return safe_load(file)
    except IsADirectoryError:
        print("ERROR: Please enter correct config file name!!")
        exit(1)


def getRootPath():
    """This function gets the root dir path.
    """
    return Path.cwd().parent.parent


def checkCreateDir(param_obj=None):
    """
    This function checks whether the dir exists or not. If it does not exist, it will create a new directory. In addition, it will also sort the dirs date-wise and increment plot foldernames to avoid overwrite. 

    Parameters
    ----------
    param_obj : dict
        param dict imported from the .yaml file, by default None
    """
    try:
        timestamp = Path(datetime.now().strftime("%Y-%m-%d"))
        base_dir = Path(param_obj['filepath']['fig_save_path'])
        base_name = param_obj['data_param']['subject_code'] + "_plot"
        
        dir_path = base_dir / timestamp
        dir_path.mkdir(parents=True, exist_ok=True)

        existing_dirs = [d for d in dir_path.iterdir() if d.is_dir() and d.name.startswith(base_name)]
        numbers = [int(d.name[len(base_name):]) for d in existing_dirs if d.name[len(base_name):].isdigit()]
        next_number = max(numbers) + 1 if numbers else 1
        dir_path = dir_path / f"{base_name}_{next_number}"

        dir_path.mkdir(parents=True)
        return dir_path

    except KeyError:
        print("ERROR!! Please ensure you pass non-empty dict object!")
        exit(1)


def createReadme(param_obj=None, dir_path=None):
    """
    This function creates a readme file in the same dir as the plots with all necessary hyperparameters of the model.

    Parameters
    ----------
    param_obj : dict
        param dict imported from the .yaml file, by default None
    dir_path: pathlib Path obj
        path for the readme file
    """
    if dir_path is None:
        print("WARNING: No dir_path specified! Creating readme file in the fig_save_path...")
        dir_path = Path(param_obj['filepath']['fig_save_path'])
    
    try:
        readme_file = dir_path / "readme.txt"
        with readme_file.open("w") as f:
            f.write(f"Scenario: {param_obj['data_param']['weights'], param_obj['data_param']['mov_type']}\n") 
            f.write(f"Subject Code: {param_obj['data_param']['subject_code']}\n")
            f.write(f"Batch Size: {param_obj['model_param']['batch_size']}\n")
            f.write(f"Epochs: {param_obj['model_param']['n_epochs']}")
            f.write(f"Feature Selection: {param_obj['preprocess_param']['feature_select']}\n")
            f.write(f"Window Size X: {param_obj['preprocess_param']['window_size_x']}\n")
            f.write(f"Window Size Y: {param_obj['preprocess_param']['window_size_y']}\n")
            f.write(f"Window Step: {param_obj['preprocess_param']['window_step']}\n")
            f.write(f"\n")
            f.write(f"HPF Filter: {param_obj['preprocess_param']['f_cutoff_hpf']}Hz\n")
            f.write(f"Variance Filter width: {param_obj['preprocess_param']['var_filter_width']}\n")
            f.write(f"MVC: {param_obj['preprocess_param']['mvc']}\n")
            f.write(f"LPF Filter: {param_obj['preprocess_param']['f_cutoff_lpf']}Hz\n")
            f.write(f"Force Activation: {param_obj['preprocess_param']['act_delay'], param_obj['preprocess_param']['act_beta1'], param_obj['preprocess_param']['act_beta2'], param_obj['preprocess_param']['act_gamma'], param_obj['preprocess_param']['act_A']}\n")
            f.write(f"\n")
            f.write(f"BPNN Neurons: {param_obj['model_param']['neurons_h1'], param_obj['model_param']['neurons_h2']}\n")
            f.write(f"BPNN Act functions: {param_obj['model_param']['act_inp'], param_obj['model_param']['act_h1'], param_obj['model_param']['act_h2']}\n")
            f.write(f"Early Stopping: {param_obj['model_param']['is_early_stop']}\n")
            f.write(f"Post Processing Filter type: {param_obj['post_train_param']['filter_type']}\n")
            f.write(f"Post Processing Filter size: {param_obj['post_train_param']['filter_size']}\n")
    
    except KeyError:
        print("ERROR!! Please ensure you pass non-empty dict object!")
        exit(1)
    
    finally:
        f.close()


def plotResults(data_ref=[], label_ref="real", data_out=[], label_out="predicted", title="", xlabel="Time in s", ylabel="", is_grid_on=True):
    """
    This function plots the result of the BPNN model

    Parameters
    ----------
    data_ref : array
        array of reference data
    label_ref : str, optional
        label for ref legend, by default "real"
    data_out : array
        array of predicted data
    label_out : str, optional
        label for predicted data legend, by default "predicted"
    title : str, optional
        title for the plot, by default ""
    xlabel : str, optional
        xlabel for the plot, by default ""
    ylabel : str, optional
        ylabel for the plot, by default ""
    is_grid_on : bool, optional
        boolean to decide grid lines visibility, by default True
    """
    plt.figure()

    x_samples = np.arange(0, len(data_ref),1)
    plt.plot(x_samples, data_ref, ls="dashed", label=label_ref)
    plt.plot(x_samples, data_out, label=label_out)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if is_grid_on:
        plt.grid()


