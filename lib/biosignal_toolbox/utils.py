# *********************************************************************************
# ************************* Imports ***********************************************
# *********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from yaml import safe_load
from datetime import datetime
from types import SimpleNamespace
import warnings
warnings.filterwarnings('ignore')

# *********************************************************************************
# ************************* Functions *********************************************
# *********************************************************************************

def loadConfig(filename=''):
    """
    This function safely loads the yaml configuration file into a dictionary.
    It also converts the dict to a Namespace object wherein keys become attributes and can be accessed by '.' operator.

    Parameters
    ----------
    filename : str
        name of the config file
    
    Author
    ------
    Author: Kartik Chari \n
    Last changed: 21.08.2025 (by Kartik Chari)
    """
    try:
        project_root = getProjectRoot()
        config_dir = project_root / 'config'
        with open(config_dir / filename, 'r') as file:
            dict_obj = safe_load(file)
        return convertDictToNamespace(dict_obj)
    except IsADirectoryError:
        print("ERROR: Please enter correct config filename!!")
        exit(1)

def convertDictToNamespace(data_inp):
    if isinstance(data_inp, dict):
        return SimpleNamespace(**{k: convertDictToNamespace(v) for k, v in data_inp.items()})
    elif isinstance(data_inp, list):
        return [convertDictToNamespace(i) for i in data_inp]
    else:
        return data_inp


def getProjectRoot(marker=""):
    """
    This function returns the project root folder path

    Parameters
    -----
    marker: str, optional
        The dir name that helps to looks for the root folder in any project structure, by default "".

    Author
    ------
    Author: Kartik Chari \n
    Last changed: 28.08.2025 (by Kartik Chari)
    """
    filepath = Path(__file__).resolve()
    if not marker:
        return filepath.parent.parent.parent
    else:
        for parent in filepath.parents:
            if parent.name == marker:
                return parent.parent
        raise RuntimeError(f"Project root not found. Looked for {marker}")


def getAbsolutePath(input_path=''):
    """
    This function ensures that the input path is absolute. It first gets the root of the project and then appends the input path to it. This function does not work with relative paths.

    Parameters
    -----
    input_path: str
        Path (wrt project root) to be made absolute, by default empty.
    Author
    ------
    Author: Kartik Chari \n
    Last changed: 27.08.2025 (by Kartik Chari)
    """
    if not input_path:
        raise ValueError("Please provide input path and/or project root path!")
    # get project root
    project_root = getProjectRoot()

    input_path = Path(input_path).expanduser()
    if input_path.is_absolute():
        try:
            # check if input_path is already inside project_root
            relative_part = input_path.resolve().relative_to(project_root)
        except ValueError:
            # break the path into pieces and strip the first '/'
            relative_part = Path(*input_path.parts[1:])
    else:
        relative_part = input_path

    return (project_root / relative_part).resolve()


def checkCreateDir(param_obj=None):
    """
    This function checks whether the dir exists or not. If it does not exist, it will create a new directory. In addition, it will also sort the dirs date-wise and increment plot foldernames to avoid overwrite. 

    Parameters
    ----------
    param_obj : dict
        param dict imported from the .yaml file, by default None
    
    Author
    ------
    Author: Kartik Chari \n
    Last changed: 13.11.2025 (by Kartik Chari)
    """
    try:
        timestamp = Path(datetime.now().strftime("%Y-%m-%d"))
        base_dir = Path(param_obj.filepath.fig_save_path)
        base_name = param_obj.data_param.subject_code + "_plot"
        
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
    
    Author
    ------
    Author: Kartik Chari \n
    Last changed: 13.11.2025 (by Kartik Chari)
    """
    if dir_path is None:
        print("WARNING: No dir_path specified! Creating readme file in the fig_save_path...")
        dir_path = Path(param_obj.filepath.fig_save_path)
    
    try:
        readme_file = dir_path / "readme.txt"
        with readme_file.open("w") as f:
            f.write(f"Scenario: {param_obj.data_param.weights, param_obj.data_param.mov_type}\n") 
            f.write(f"Subject Code: {param_obj.data_param.subject_code}\n")
            f.write(f"Batch Size: {param_obj.model_param.batch_size}\n")
            f.write(f"Epochs: {param_obj.model_param.n_epochs}")
            f.write(f"Feature Selection: {param_obj.preprocess_param.feature_select}\n")
            f.write(f"Window Size X: {param_obj.preprocess_param.window_size_x}\n")
            f.write(f"Window Size Y: {param_obj.preprocess_param.window_size_y}\n")
            f.write(f"Window Step: {param_obj.preprocess_param.window_step}\n")
            f.write(f"\n")
            f.write(f"HPF Filter: {param_obj.preprocess_param.f_cutoff_hpf}Hz\n")
            f.write(f"Variance Filter width: {param_obj.preprocess_param.var_filter_width}\n")
            f.write(f"MVC: {param_obj.preprocess_param.mvc}\n")
            f.write(f"LPF Filter: {param_obj.preprocess_param.f_cutoff_lpf}Hz\n")
            f.write(f"Force Activation: {param_obj.preprocess_param.act_delay, param_obj.preprocess_param.act_beta1, param_obj.preprocess_param.act_beta2, param_obj.preprocess_param.act_gamma, param_obj.preprocess_param.act_A}\n")
            f.write(f"\n")
            f.write(f"BPNN Neurons: {param_obj.model_param.neurons_h1, param_obj.model_param.neurons_h2}\n")
            f.write(f"BPNN Act functions: {param_obj.model_param.act_inp, param_obj.model_param.act_h1, param_obj.model_param.act_h2}\n")
            f.write(f"Early Stopping: {param_obj.model_param.is_early_stop}\n")
            f.write(f"Post Processing Filter type: {param_obj.post_train_param.filter_type}\n")
            f.write(f"Post Processing Filter size: {param_obj.post_train_param.filter_size}\n")
    
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
    
    Author
    ------
    Author: Kartik Chari \n
    Last changed: 13.11.2025 (by Kartik Chari)
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


